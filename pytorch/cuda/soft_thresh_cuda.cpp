#include <torch/torch.h>
#include <vector>
#include <iostream>

// CUDA declarations
std::vector<at::Tensor> soft_thresh_fwd_cuda(
    at::Tensor input,
    at::Tensor thresh);

std::vector<at::Tensor> soft_thresh_bwd_cuda(
    at::Tensor dy, 
    at::Tensor input,
    at::Tensor gain,
    at::Tensor mag); 


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// C++ wrapper
std::vector<at::Tensor> soft_thresh_fwd(
    at::Tensor input,
    at::Tensor thresh) {
    CHECK_INPUT(input);
    CHECK_INPUT(thresh);
    return soft_thresh_fwd_cuda(input, thresh);
}

std::vector<at::Tensor> soft_thresh_bwd(
    at::Tensor dy, 
    at::Tensor input,
    at::Tensor gain,
    at::Tensor mag) {
    CHECK_INPUT(dy);
    CHECK_INPUT(input);
    CHECK_INPUT(gain);
    CHECK_INPUT(mag);
    return soft_thresh_bwd_cuda(dy, input, gain, mag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &soft_thresh_fwd, "Soft Thresholding forward (CUDA)");
  m.def("backward", &soft_thresh_bwd, "Soft Thresholding backward (CUDA)");
}

