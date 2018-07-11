""" This is the same code as example 1, however it uses unified memory
"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
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
    # Cuda code for unified memory:
    #   1. Allocate memory to be device and host available
    #   2. Call the parallelizable kernel with a set number of blocks and
    #       threads
    #   3. Synchronize (wait for completion)

    # Step 1
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    a = cuda.managed_empty(shape=1<<24, dtype=np.float32,
                           mem_flags=cuda.mem_attach_flags.GLOBAL)
    b = cuda.managed_empty(shape=1<<24, dtype=np.float32,
                           mem_flags=cuda.mem_attach_flags.GLOBAL)
    a[:] = np.random.randn(1<<24).astype('float32')
    b[:] = np.random.randn(1<<24).astype('float32')
    c2 = cuda.managed_zeros(shape=1<<24, dtype=np.float32,
                           mem_flags=cuda.mem_attach_flags.GLOBAL)

    start = time.time()
    c = a + b
    print('Numpy implementation took {}'.format(time.time()-start))


    # Step 2 - Create the kernel and call it
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')
    start = time.time()
    func(c2, a, b, np.int32(1<<24), grid=((1<<24)//256,1), block=(256,1,1))
    # Wait for completion
    pycuda.autoinit.context.synchronize()
    print('Gpu implementation took {}'.format(time.time()-start))

    start = time.time()
    for i in range(50):
        func(c2, a, b, np.int32(1<<24), grid=((1<<24)//256,1), block=(256,1,1))
    # Wait for completion
    pycuda.autoinit.context.synchronize()
    print('Gpu implementation repeated took on average {}'.format((time.time()-start)/50))

    # Synchronize (copy back)
    np.testing.assert_array_almost_equal(c, c2)
