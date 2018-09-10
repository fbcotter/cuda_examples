CUDA Examples
=============

This repo contains code examples and explanations on how to accelerate some code
using CUDA. The repo was prepared for a talk I gave to the Signal Processing
Group in July 2018.

There are many CUDA examples on the internet, these examples are not too
different, just framed from the point of view that may help the SigProc group
the most. In particular - calling fast but perhaps strange code from python.

Using CUDA in python needs some python - gpu bridge. There are several packages
out there that do this, and I am not fully aware of the ins-and-outs of each of
them. However, those that I have come across so far are:

- Tensorflow/Pytorch - we all know these. They provide gpu accelerated code with
  automatic differentiation. Don't really let you write CUDA code unless you
  want to extend them, which is difficult.
- cupy: Designed to be exactly like numpy but to run on gpus. For the most part,
  works like pytorch/tf, except it does have the ability to compile cuda code.
- pycuda: probably the most natural one if you want to write most of your code
  in python and write some simple cuda kernels.
- numba/pyculib: I don't think this lets you write CUDA kernel code, however it
  does give access to all the low level cuda libraries (cuFFT, cuSPARSE,
  cuBLAS). I think it accepts numpy arrays/numba cuda arrays and transfers the
  memory across. I'd like to look more at this, my guess is numba uses it to do
  their gpu acceleration.

In these examples, we start with pycuda, then use a bit of cupy to compile
kernels that can work with pytorch.

Example 1
---------
This is based off `this blog`__, which gives a great introduction to CUDA's
thread and memory management.

__ https://devblogs.nvidia.com/even-easier-introduction-cuda/

