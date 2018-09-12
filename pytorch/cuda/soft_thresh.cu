#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_THREADS 1024

// CUDA main functions
template <typename scalar_t>
__global__ void soft_thresh_cuda_fwd_kern(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ din,
    const scalar_t* __restrict__ input,
    const scalar_t* thresh,
    int64_t size) {

  // Use the grid loop
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x);
     i < size; 
     i += blockDim.x * gridDim.x) {
    scalar_t delta = max(abs(input[i]) - *thresh, scalar_t(0.0));
    output[i] = copysign(delta, input[i]);
    din[i] = copysign((scalar_t)(delta > scalar_t(0.0)), input[i]);
  }
}

template <typename scalar_t>
__global__ void soft_thresh_cuda_bwd_kern(
    scalar_t* __restrict__ dx,
    scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ din,
    int64_t size) {

  // Use the grid loop
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x);
     i < size; 
     i += blockDim.x * gridDim.x) {
    const scalar_t pos = (scalar_t)(dx[i] != scalar_t(0.0));
    dx[i] = dy[i] * pos;
    dt[i] = dy[i] * dx[i];
  }
}


// CUDA wrapper functions
std::vector<at::Tensor> soft_thresh_fwd_cuda(
    at::Tensor input,
    at::Tensor thresh) {

  auto output = at::zeros_like(input);
  auto din = at::zeros_like(input);
  int64_t size = at::numel(input);
  int blocks = 1000;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "soft_thresh_cuda", ([&] {
    soft_thresh_cuda_fwd_kern<scalar_t> <<<blocks, NUM_THREADS>>> (
        output.data<scalar_t>(),
        din.data<scalar_t>(),
        input.data<scalar_t>(),
        thresh.data<scalar_t>(),
        size);
  }));
  return {output, din};
}


std::vector<at::Tensor> soft_thresh_bwd_cuda(
    at::Tensor dy, 
    at::Tensor din) {

  auto dx = at::zeros_like(dy);
  auto dt = at::zeros_like(dy);
  int64_t size = at::numel(dy);
  int blocks = 1000;

  AT_DISPATCH_FLOATING_TYPES(dy.type(), "soft_thresh_cuda", ([&] {
    soft_thresh_cuda_bwd_kern<scalar_t><<<blocks, NUM_THREADS>>>(
        dx.data<scalar_t>(),
        dt.data<scalar_t>(),
        dy.data<scalar_t>(),
        din.data<scalar_t>(),
        size);
  }));
  return {dx, at::sum(dt)};
}
