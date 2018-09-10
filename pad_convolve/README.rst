Pad Convolve
============

These examples examine offloading the padding to the CUDA kernel. Currently in
pytorch and tensorflow, convolution only works with zero padding. You can call
a function beforehand that pads the input, then do zero padding afterwads, but
this is slow. In pytorch things are worse, as the function doesn't even do
symmetric padding.

scipy_convolve compares the time to run convolution in scipy vs in our custom
cuda kernel.

torch_convolve does the same for pytorch
