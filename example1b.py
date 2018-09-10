""" Example 1b

This is the same task as example 1, however it uses a grid block loop - the
kernel now has a loop in it rather than a single operation.

This is useful in case we support any problem size, even if it is larger than
the number of blocks the CUDA device supports - note that the call to func
doesn't need to work out the correct size anymore.
"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit  # noqa - need this to initialize gpus
from pycuda.compiler import SourceModule

kernel = """
extern "C"
__global__ void mykern(float *dst, const float *a, const float *b, int N)
{
    for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        dst[i] = a[i] + b[i];
    }
}
"""

if __name__ == '__main__':
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    a = np.random.randn(1 << 24).astype('float32')
    b = np.random.randn(1 << 24).astype('float32')
    c2 = np.zeros(1 << 24, dtype='float32')

    start = time.time()
    c = a + b
    np_time = time.time() - start
    print('Numpy implementation took {}'.format(np_time))

    # Step 1
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(b.nbytes)

    # Step 2
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Step 3 - Create the kernel and call it
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')
    start = time.time()
    func(c_gpu, a_gpu, b_gpu, np.int32(1 << 24), grid=(32,1,1), block=(256,1,1))

    # Synchronize (copy back)
    cuda.memcpy_dtoh(c2, c_gpu)
    cuda_time = time.time() - start
    np.testing.assert_array_almost_equal(c, c2)

    # Free memory
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()

    print('Gpu total implementation took {}'.format(time.time()-start))
    print('Speedup: {:.2f}x'.format(np_time/cuda_time))
