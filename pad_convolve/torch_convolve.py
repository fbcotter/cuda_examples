""" This is the same code as example 1, however it uses unified memory
"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
import torch
import torch.nn.functional as F


rowfilter_kernel = """
extern "C"
__global__ void rowfilter(
float* dest, const float* src, const float *w,
  int N, int C, int H, int W, int M) {
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


def reflect(x, minx, maxx):
    """ Do symmetric padding for numpy """
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
    a = np.random.randn(16, 3, 256, 256).astype('float32')
    w = np.random.randn(1, 11).astype('float32')

    print('''Convolving an input of shape {} with a filter of shape {} in torch
and cuda (note that this is 2d convolution. the first 2 dimensions of the
input are batch and channel dimensions\n'''.format(a.shape, w.shape))

    a_t = torch.tensor(a).cuda()
    w_t = np.reshape(w[:,::-1], [1, 1, *w.shape])
    w_t = np.repeat(w_t, repeats=3, axis=0)
    w_t = np.copy(w_t)
    w_t = torch.tensor(w_t, dtype=torch.float32).cuda()
    m = w_t.shape[3] // 2
    c = a.shape[3]
    xe = reflect(np.arange(-m, c+m, dtype='int32'), -0.5, c-0.5)

    # Run once to 'burn in'
    y_t = F.conv2d(a_t[:,:,:,xe], w_t, groups=3)
    torch.cuda.synchronize()
    start = time.time()
    y_t = F.conv2d(a_t[:,:,:,xe], w_t, groups=3)
    torch.cuda.synchronize()
    torch_time = time.time() - start

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
    start = time.time()
    func(y_gpu, a_gpu, w_gpu, np.int32(a.shape[0]), np.int32(a.shape[1]),
         np.int32(a.shape[2]), np.int32(a.shape[3]), np.int32(m),
         grid=(64,1,1), block=(1024,1,1))
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    # Synchronize (copy back)
    y2 = np.zeros_like(a).astype('float32')
    cuda.memcpy_dtoh(y2, y_gpu)
    np.testing.assert_array_almost_equal(y_t, y2, decimal=4)

    print('Torch implementation took {:.5f}s'.format(torch_time))
    print('Gpu kernel implementation took {:.5f}s'.format(cuda_time))
    print('Speedup: {:.2f}x'.format(torch_time/cuda_time))
    # Free memory
    a_gpu.free()
    w_gpu.free()
    y_gpu.free()
