Extending Pytorch with CUDA
===========================

These examples attempt to extend pytorch with CUDA. In this example, we will try
do a soft thresholding function in pytorch. We compare the runtimes when we do
it directly in torch vs in cpp vs in cuda.

You can test the results yourself by running benchmark.py. You'll need to build
the cpp and cuda implementations first. To do this, change into each directory
and run

.. code:: bash

  python setup.py install

It should create a `build/lib.linux-x86_64-3.6/` or similar folder.

Once you have done this, check that the implementations give the same results by
running:

.. code:: bash

  python soft_thresh_test.py

Finally, you can compare the runtimes of the python, cpp and cuda by running

.. code:: bash
  
  python benchmark.py py [-c]
  python benchmark.py cpp [-c]
  python benchmark.py cuda

Where the optional `-c` flag runs the python/cpp code on the gpu (automatically
converting it for us).

The results are impressive! The naive python cpu implementation is::

  Forward: 689.983/817.256 us | Backward 1421.690/1886.859 us

The cpp implementation of was mostly about the same::

  Forward: 726.700/958.605 us | Backward 1430.273/2345.104 us

And on the gpus, the torch implementation::

  Forward: 228.167/329.758 us | Backward 561.237/723.530 us

CPP gpu implementation::

  Forward: 286.341/340.600 us | Backward 637.770/723.438 us

CUDA implementation::

  Forward: 76.056/98.617 us | Backward 269.175/315.604 us
