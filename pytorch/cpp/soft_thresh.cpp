#include <torch/torch.h>
#include <vector>
#include <iostream>

// dx = 
at::Tensor dx_soft_thresh(
    at::Tensor y, 
    at::Tensor input, 
    at::Tensor gain, 
    at::Tensor mag) {
  auto pos = at::_cast_Float(gain > 0.0);
  return y * (gain + at::pow(input, 2)/at::pow(mag, 2) * (pos - gain));
}

at::Tensor dt_soft_thresh(
    at::Tensor y, 
    at::Tensor input, 
    at::Tensor gain,
    at::Tensor mag) {
  auto pos = at::_cast_Float(gain > 0.0);
  return at::sum(y * -(input/mag *pos));
}

std::vector<at::Tensor> soft_thresh_fwd(
    at::Tensor input,
    at::Tensor thresh) {
  auto mag = at::abs(input);
  auto denom = mag + at::_cast_Float(mag < thresh);
  auto gain = at::relu(mag - thresh)/denom;
  return {input * gain, mag, gain};
}

std::vector<at::Tensor> soft_thresh_bwd(
    at::Tensor dy, 
    at::Tensor input,
    at::Tensor gain,
    at::Tensor mag) {
  auto dx = dx_soft_thresh(dy, input, gain, mag);
  auto dt = dt_soft_thresh(dy, input, gain, mag);
  return {dx, dt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &soft_thresh_fwd, "Soft Thresholding forward");
  m.def("backward", &soft_thresh_bwd, "Soft Thresholding backward");
}
