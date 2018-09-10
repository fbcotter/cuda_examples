#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_THREADS 1024

// CUDA helper functions
namespace {  // make these functions only accessible in this file
template <typename scalar_t>
__device__ __forceinline__ scalar_t dx_soft_thresh(
    scalar_t dy, 
    scalar_t input, 
    scalar_t gain, 
    scalar_t mag) {
  const scalar_t pos = (scalar_t)(gain > 0);
  const scalar_t dxdy = gain + (input*input)/(mag*mag) * (pos - gain);
  return dy * dxdy;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t dt_soft_thresh(
    scalar_t dy, 
    scalar_t input, 
    scalar_t gain, 
    scalar_t mag) {
  const scalar_t pos = (scalar_t)(gain > 0);
  return dy * -(input/mag * pos);
}

// CUDA main functions
template <typename scalar_t>
__global__ void soft_thresh_cuda_fwd_kern(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ mag,
    scalar_t* __restrict__ gain,
    const scalar_t* __restrict__ input,
    const scalar_t* thresh,
    int64_t size) {

  // Use the grid loop
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x);
     i < size; 
     i += blockDim.x * gridDim.x) {
    mag[i] = abs(input[i]);
    const scalar_t below_thresh = (scalar_t) (mag[i] < *thresh);
    const scalar_t denom = mag[i] + below_thresh;
    gain[i] = max(mag[i] - *thresh, scalar_t(0.0))/denom;
    output[i] = input[i] * gain[i];
  }
}

template <typename scalar_t>
__global__ void soft_thresh_cuda_bwd_kern(
    scalar_t* __restrict__ dx,
    scalar_t* __restrict__ dt,
    scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gain,
    const scalar_t* __restrict__ mag,
    int64_t size) {

  // Use the grid loop
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x);
     i < size; 
     i += blockDim.x * gridDim.x) {
    dx[i] = dx_soft_thresh(dy[i], input[i], gain[i], mag[i]);
    dt[i] = dt_soft_thresh(dy[i], input[i], gain[i], mag[i]);
  }
}
} // namespace


// CUDA wrapper functions
std::vector<at::Tensor> soft_thresh_fwd_cuda(
    at::Tensor input,
    at::Tensor thresh) {

  auto output = at::zeros_like(input);
  auto mag = at::zeros_like(input);
  auto gain = at::zeros_like(input);
  int64_t size = at::numel(input);
  int blocks = 8;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "soft_thresh_cuda", ([&] {
    soft_thresh_cuda_fwd_kern<scalar_t> <<<blocks, NUM_THREADS>>> (
        output.data<scalar_t>(),
        mag.data<scalar_t>(),
        gain.data<scalar_t>(),
        input.data<scalar_t>(),
        thresh.data<scalar_t>(),
        size);
  }));
  return {output, mag, gain};
}


std::vector<at::Tensor> soft_thresh_bwd_cuda(
    at::Tensor dy, 
    at::Tensor input,
    at::Tensor gain,
    at::Tensor mag) {

  auto dx = at::zeros_like(input);
  auto dt = at::zeros_like(input);
  int64_t size = at::numel(input);
  int blocks = 8;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "soft_thresh_cuda", ([&] {
    soft_thresh_cuda_bwd_kern<scalar_t><<<blocks, NUM_THREADS>>>(
        dx.data<scalar_t>(),
        dt.data<scalar_t>(),
        dy.data<scalar_t>(),
        input.data<scalar_t>(),
        gain.data<scalar_t>(),
        mag.data<scalar_t>(),
        size);
  }));
  return {dx, at::sum(dt)};
}
