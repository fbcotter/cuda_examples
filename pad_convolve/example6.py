""" This is the same code as example 1, however it uses unified memory
"""
import numpy as np
import time
import cupy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import torch
import torch.nn.functional as F
from torch.autograd import Function
from scipy.signal import convolve2d
from collections import namedtuple
from string import Template
import py3nvml
py3nvml.grab_gpus(1)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
Stream = namedtuple('Stream', ['ptr'])

CUDA_NUM_THREADS = 1024


rowfilter_kernel = """
extern "C"
__global__ void rowfilter(
float* dest, const float* src, const float *w, int N, int C, int H, int W, int M) {
/* dest - output array. should be same shape as input
   src - input array
   w - input kernel. Should be a 1d array
   N, C, H, W - input tensor sizes
   M - weight size
   zero_m - position of the 'zero' in the weight. As python doesn't support
       negative indexing.
*/
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N*C*H*W; i += blockDim.x * gridDim.x) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;

        // The stride of the input assuming correct padding - if the filter has
        // even length, we will pad M/2 on either side, so the stride will be
        // W + M. If it has odd length, we will padd (M-1)/2 either side, so
        // the stride will be W+M-1.
        const int S = W + M - (M % 2);
        float value = 0;
#pragma unroll
        for (int kw = 0; kw < M; kw++) {
            const int offset = n*C*H*S + c*H*S + y*S + x + kw;
            value += w[M-1-kw]*src[offset];
        }
        dest[i] = value;
    }
}
"""
rowfilter_pad_kernel2 = """
extern "C"
__global__ void rowfilter_pad(float *dest, const float* src, const float* w,
                              int N, int C, int H, int Wmin, int Wmax,
                              int W_in, int Mlow, int Mhigh, int rev,
                              int stride)
/*
    We want to avoid costly padding operations, but instead just pull from x
    modulo. We also allow the output to be an arbitrary width, not necessarily
    going from 0 to W_in. For support outside [0, W_in) symmetric padding is
    used on the input.

    Wmin - low index of the output x (can be negative)
    Wmax - high index of the output x (can be larger than W_in)
    W_in - the width of the input.
    Mlow - low index of the filter (can be negative)
    Mhigh - high index of the filter
*/
{
    const int W = Wmax - Wmin + 1;
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
        i < N*C*H*W; i += stride*(blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        for (int k = Mlow; k <= Mhigh; k++) {
            int x_in = x + Wmin + k;

            // handle padding - the below complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = x_in >= 0 ? (x_in / W_in) % 2 :
                                          1 - (((-x_in-1)/W_in) % 2);
            const int res = (x_in % W_in + W_in) % W_in;
            x_in = (group == 1) ? (W_in-1) - res : res;

            const int offset = n*C*H*W_in + c*H*W_in + y*W_in + x_in;
            value += rev ? w[Mhigh-k] * src[offset] : w[k-Mlow] * src[offset];
        }
     dest[i / stride] = value;
     }
}
"""
rowfilter_pad_kernel = """
extern "C"
__global__ void rowfilter_pad(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Win, int Mlow, int Mhigh, int rev, int stride) {
/* dest - output array. should be same shape as input
   src - input array
   w - input kernel. Should be a 1d array
   N, C, H, W - input tensor sizes
   Mlow - idx of most negative filter tap
   Mhigh - idx of most positive filter tap
   rev - used for calculating gradients - need to do correlation, and
    some funny things with the filter.
*/
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        // Use convolution formula: y[n] = sum h[k]*x[n-k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int x_in = x - k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = x_in >= 0 ? ((x_in / Win) % 2)
                                        : 1-(((-x_in-1)/Win) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (x_in % Win + Win) % Win;
            x_in = (group == 1) ? (Win-1) - res : res;

            const int offset = n*C*H*Win + c*H*Win + y*Win + x_in;
            if (rev) {
                value += w[Mhigh-k] * src[offset];
            } else {
                value +=  w[k-Mlow] * src[offset];
            }
        }
        dest[i/stride] = value;
    }
}
"""

rowfilter_pad_kernel_bwd = """
extern "C"
__global__ void rowfilter_pad_bwd(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Win, int Mlow, int Mhigh, int stride) {
/* dest - output array. should be same shape as input
   src - input array
   w - input kernel. Should be a 1d array
   N, C, H, W - input tensor sizes
   Mlow - idx of most negative filter tap
   Mhigh - idx of most positive filter tap
   rev - used for calculating gradients - need to do correlation, and
    some funny things with the filter.
*/
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        // Use correlation formula: y[n] = sum h[k]*x[n+k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int x_in = x + k;
            int k_in = (x_in < 0 || x_in >= Win) ? -k : k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = x_in >= 0 ? ((x_in / Win) % 2)
                                        : 1-(((-x_in-1)/Win) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (x_in % Win + Win) % Win;
            x_in = (group == 1) ? (Win-1) - res : res;

            const int offset = n*C*H*Win + c*H*Win + y*Win + x_in;
            value += w[k_in - Mlow] * src[offset];
        }
        dest[i/stride] = value;
    }
}
"""

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


class RowFilter(Function):

    def __init__(self):
        super(RowFilter, self).__init__()

    def forward(self, input, weight):
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda
        n, ch, h, w = input.shape
        kh, kw = weight.shape
        m2 = kw // 2
        output = torch.zeros_like(input)
        xe = reflect(np.arange(-m2, w+m2, dtype='int32'), -0.5, w-0.5)
        input = input[:,:,:,xe]

        with torch.cuda.device_of(input):
            f = load_kernel('rowfilter', rowfilter_kernel)
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(128,1,1),
              args=[output.data_ptr(), input.data_ptr(), weight.data_ptr(),
                    np.int32(n), np.int32(ch), np.int32(h), np.int32(w),
                    np.int32(kw)],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

class RowFilter_pad(Function):

    def __init__(self, weight, klow, khigh):
        super(RowFilter_pad, self).__init__()
        self.weight = weight
        self.klow = klow
        self.khigh = khigh
        assert abs(klow) == khigh
        self.delay = (khigh + klow) / 2
        self.pad_end = 1 - ((khigh - 1 - klow) % 2)
        self.f = load_kernel('rowfilter_pad', rowfilter_pad_kernel)
        self.fbwd = load_kernel('rowfilter_pad_bwd', rowfilter_pad_kernel_bwd)

    #  @staticmethod
    def forward(ctx, input):
        assert input.dim() == 4 and input.is_cuda and ctx.weight.is_cuda
        n, ch, h, w = input.shape
        ctx.in_shape = (n, ch, h, w)
        output = torch.zeros((n, ch, h, w + ctx.pad_end),
                             dtype=torch.float32,
                             requires_grad=input.requires_grad).cuda()

        with torch.cuda.device_of(input):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[output.data_ptr(), input.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w+ctx.pad_end), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(0),
                        np.int32(1)],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    #  @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.in_shape
        n, ch, h, w = grad_out.shape
        grad_input = torch.zeros(in_shape, dtype=torch.float32).cuda()
        with torch.cuda.device_of(grad_out):
            ctx.fbwd(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[grad_input.data_ptr(), grad_out.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w-ctx.pad_end), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input


if __name__ == '__main__':
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    #  a = np.random.randn(16, 3, 64, 64).astype('float32')
    w = np.random.randn(1, 11).astype('float32')

    # Define an input 'tensor' a and a rowfilter w.
    a = np.repeat(np.expand_dims(np.arange(10),axis=0), repeats=5,
                  axis=0).astype('float32')
    a = np.repeat(np.expand_dims(a, axis=0), repeats=3, axis=0)
    a = np.repeat(np.expand_dims(a, axis=0), repeats=2, axis=0)
    #  w = np.array([[1,2,3,4,5]]).astype('float32')

    # Push a to the gpu and make w a 4-dimensional tensor, ready for torch's
    # conv2d function (i.e. need to add some dimensions and transpose it)
    a_t = torch.tensor(a, requires_grad=True)
    a_t_gpu = a_t.cuda()
    w_t = np.reshape(w[:,::-1], [1, 1, *w.shape])
    w_t = np.repeat(w_t, repeats=3, axis=0)
    w_t = np.copy(w_t)
    w_t = torch.tensor(w_t, dtype=torch.float32).cuda()

    # Torch doesn't do symmetric padding, so do that manually by making a new
    # tensor that is larger
    m = w_t.shape[3]
    m2 = m // 2
    c = a.shape[3]
    xe = reflect(np.arange(-m2, c+m2, dtype='int32'), -0.5, c-0.5)

    # Now do torch's convolution
    # Run once to 'burn in'
    y_t = F.conv2d(a_t_gpu[:,:,:,xe], w_t, groups=3)
    start = time.time()
    for i in range(10):
        y_t = F.conv2d(a_t_gpu[:,:,:,xe], w_t, groups=3)
    print('Torch implementation took on avg (10 runs):\t{}'.format((time.time()-start)/10))
    y_t.backward(torch.ones_like(y_t))
    grad1 = np.array(a_t.grad.data)
    a_t.grad.data.zero_()

    # In comparison, do the convolution directly with our above code.
    #
    # The first of these also manually extends a to be larger. The second does
    # the symmetric padding in the low-level cuda code and it turns out this is
    # faster.
    w_t2 = torch.tensor(w, dtype=torch.float32).cuda()
    mod = RowFilter()
    y_t2 = mod(a_t_gpu, w_t2)
    start = time.time()
    for i in range(10):
        y_t2 = mod(a_t_gpu, w_t2)
    print('My implementation took on avg (10 runs):\t{}'.format((time.time()-start)/10))


    mod = RowFilter_pad(w_t2, -m2+(1-m%2), m2)
    y_t3 = mod(a_t_gpu)
    start = time.time()
    for i in range(10):
        y_t3 = mod(a_t_gpu)
    print('My implementation took on avg (10 runs):\t{}'.format((time.time()-start)/10))
    y_t3.backward(torch.ones_like(y_t3))
    grad2 = np.array(a_t.grad.data)
    a_t.grad.data.zero_()

    y_fwd = mod(torch.ones_like(a_t_gpu))
    #  np.testing.assert_array_almost_equal(y_t.data.detach().numpy(), y_t3.data.detach().numpy(), decimal=4)
    np.testing.assert_array_almost_equal(y_t.cpu().detach().numpy(), y_t3.cpu().detach().numpy(), decimal=4)
    np.testing.assert_array_almost_equal(grad1, grad2, decimal=4)
    from torch.autograd import gradcheck
    gradcheck(mod, (a_t_gpu,), eps=1e-2, atol=1e-3)
