""" Example 1

This is a simple example that shows how to add two very large arrays.
It compares the run time to numpy.

Cuda steps:

1. Allocate memory on device
2. Fill memory with data (e.g. copy data from host to device, fill with
     random numbers, etc.)
3. Call the parallelizable kernel with a set number of blocks and
     threads
4. Synchronize (wait for completion) and copy data back from device
5. Free memory

"""
import numpy as np
import time
import pycuda.autoinit  # noqa - Have to import otherwise won't run.
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

kernel = """
extern "C"
__global__ void mykern(float *dst, const float *a, const float *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dst[i] = a[i] + b[i];
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
    print('Numpy implementation took {:.5f}s'.format(np_time))

    # Step 1 - Allocate memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(b.nbytes)

    # Step 2 - Fill memory
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Step 3 - Create the kernel and call it
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')
    start = time.time()
    func(c_gpu, a_gpu, b_gpu, np.int32(1 << 24),
         grid=((1 << 24)//256,1), block=(256,1,1))

    # Step 4 - Synchronize (copy back)
    cuda.memcpy_dtoh(c2, c_gpu)
    cuda_time = time.time() - start

    # Step 5 - Free memory
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()

    print('Gpu total took {:.5f}s'.format(cuda_time))
    print('Speedup: {:.2f}x'.format(np_time/cuda_time))
    np.testing.assert_array_almost_equal(c, c2)
