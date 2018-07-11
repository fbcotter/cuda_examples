""" This is the same code as example 1, however it uses unified memory
"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
      i < (n);
      i += blockDim.x * gridDim.x)
'''

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
        for (int kw = 0; kw < M; kw++) {
            const int offset = n*C*H*S + c*H*S + y*S + x + kw;
            value += w[M-1-kw]*src[offset];
        }
        dest[i] = value;
    }
}
"""

'''
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const ${Dtype}* weight = weight_data + c * ${kernel_h} * ${kernel_w};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}
'''


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


if __name__ == '__main__':
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    #  a = np.random.randn(16, 3, 256, 256).astype('float32')
    a = np.repeat(np.expand_dims(np.arange(10),axis=0), repeats=5,
                  axis=0).astype('float32')
    a = np.repeat(np.expand_dims(a, axis=0), repeats=3, axis=0)
    a = np.repeat(np.expand_dims(a, axis=0), repeats=2, axis=0)
    w = np.random.randn(1, 5).astype('float32')
    #  w = np.array([[1, 1, 1,  1, 1]]).astype('float32')

    y = np.zeros_like(a)
    for i in range(2):
        for j in range(3):
            a[i,j] += i*2 + j
            y[i,j] = convolve2d(a[i,j], w, mode='same', boundary='symm')

    # Cuda code:
    # Typical steps are
    #   1. Allocate memory on device
    #   2. Fill memory with data (e.g. copy data from host to device, fill with
    #       random numbers, etc.)
    #   3. Call the parallelizable kernel with a set number of blocks and
    #       threads
    #   4. Synchronize (wait for completion) and copy data back from device
    #   5. Free memory

    # Step 1
    c = a.shape[-1]
    m = w.shape[-1]
    m2 = m // 2
    xe = reflect(np.arange(-m2, c+m2, dtype='int32'), -0.5, c - 0.5)
    a2 = np.copy(a[:,:,:,xe])
    a_gpu = cuda.mem_alloc(a2.nbytes)
    w_gpu = cuda.mem_alloc(w.nbytes)
    y_gpu = cuda.mem_alloc(a.nbytes)

    #  # Step 2
    cuda.memcpy_htod(a_gpu, np.ascontiguousarray(a2))
    cuda.memcpy_htod(w_gpu, w)

    #  # Step 3 - Create the kernel and call it
    mod = SourceModule(rowfilter_kernel)
    func = mod.get_function('rowfilter')
    #  start2 = time.time()
    start2 = time.time()
    func(y_gpu, a_gpu, w_gpu, np.int32(2), np.int32(3), np.int32(5), np.int32(c), np.int32(m),
         grid=(32,1,1), block=(10,1,1))
    #  func(c_gpu, a_gpu, b_gpu, np.int32(1<<24), grid=(32,1,1), block=(256,1,1))
    print('Gpu kernel implementation took {}'.format(time.time()-start2))

    # Synchronize (copy back)
    y2 = np.zeros_like(a).astype('float32')
    cuda.memcpy_dtoh(y2, y_gpu)
    np.testing.assert_array_almost_equal(y, y2, decimal=5)

    # Free memory
    a_gpu.free()
    w_gpu.free()
    y_gpu.free()

