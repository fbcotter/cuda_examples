""" Example 2

Now we show how to use cupy
"""
import numpy as np
import time
import cupy as cp

if __name__ == '__main__':
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    a = np.random.randn(1 << 24).astype('float32')
    b = np.random.randn(1 << 24).astype('float32')

    start = time.time()
    c = a + b
    np_time = time.time() - start
    print('Numpy implementation took {:.5}s'.format(time.time()-start))

    # Cupy code - much simpler
    a_gpu = cp.array(a)
    b_gpu = cp.array(b)

    start = time.time()
    c_gpu = a_gpu + b_gpu
    cupy_time = time.time() - start
    print('Cupy implementation took {:.5}s'.format(time.time() - start))
    print('Speedup: {:.2f}x'.format(np_time/cupy_time))
