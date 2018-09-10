""" Soft Thresholding

In this example, we compare doing soft thresholding in pytorch directly, vs
using c++ code.
"""
import torch
import sys
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import os
build_dir = ''
for dr in os.listdir('cpp/build/'):
    if dr.startswith('lib'):
        build_dir = os.path.join('cpp/build', dr)
if build_dir == '':
    raise ValueError('Could not find the built binaries. Have you built them?')
sys.path.insert(0, build_dir)
build_dir = ''
for dr in os.listdir('cuda/build/'):
    if dr.startswith('lib'):
        build_dir = os.path.join('cuda/build', dr)
if build_dir == '':
    raise ValueError('Could not find the built binaries. Have you built them?')
import soft_thresh  # noqa
import soft_thresh_cuda  # noqa


# ########### Torch Implementation #############
class SoftShrink1(Function):
    @staticmethod
    def forward(ctx, x1, t1):
        m1 = torch.abs(x1)
        # When the magnitude is below the threshold, add an offset
        denom = m1 + (m1 < t1).float()
        gain = torch.relu(m1 - t1)/denom
        ctx.save_for_backward(x1, gain, m1)
        ctx.complex = complex
        return x1 * gain

    @staticmethod
    def backward(ctx, grad_y):
        x1, gain1, m1 = ctx.saved_tensors
        grad_x1 = None
        grad_t1 = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_y * (gain1 + x1**2/m1**2 *
                                ((gain1 > 0).float() - gain1))

        if ctx.needs_input_grad[1]:
            grad_t1 = torch.sum(grad_y * (x1/m1 * -1 * (gain1 > 0).float()))

        return grad_x1, grad_t1


class SoftShrink(nn.Module):
    def __init__(self, t_init=1.0, complex=False):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t_init).float())

    def forward(self, x):
        """ Applies Soft Thresholding to x """
        return SoftShrink1.apply(x, self.t)


# ############## C++ Implementation ###############
class SoftShrinkC1(Function):
    @staticmethod
    def forward(ctx, x1, t1):
        y, m, gain = soft_thresh.forward(x1, torch.tensor(t1))
        ctx.save_for_backward(x1, gain, m)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gain, m = ctx.saved_tensors
        grad_x = None
        grad_t = None
        dx, dt = soft_thresh.backward(grad_y, x, gain, m)
        if ctx.needs_input_grad[0]:
            grad_x = dx

        if ctx.needs_input_grad[1]:
            grad_t = dt

        return grad_x, grad_t


class SoftShrinkC(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        return SoftShrinkC1.apply(x, self.t)


# ############## CUDA Implementation ###############
class SoftShrinkCUDA1(Function):
    @staticmethod
    def forward(ctx, x1, t1):
        y, m, gain = soft_thresh_cuda.forward(x1, torch.tensor(t1))
        ctx.save_for_backward(x1, gain, m)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gain, m = ctx.saved_tensors
        grad_x = None
        grad_t = None
        dx, dt = soft_thresh_cuda.backward(grad_y, x, gain, m)
        if ctx.needs_input_grad[0]:
            grad_x = dx

        if ctx.needs_input_grad[1]:
            grad_t = dt

        return grad_x, grad_t


class SoftShrinkCUDA(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        return SoftShrinkCUDA1.apply(x, self.t)


# ############ Test Implementations ##############
def test_equal():
    S1 = SoftShrink(.5)
    S2 = SoftShrinkC(.5)
    S3 = SoftShrinkCUDA(.5).cuda()
    X = torch.randn(10,10,5, requires_grad=True)

    y1 = S1(X)
    y1_np = np.array(y1.detach().data)
    y1.backward(torch.ones_like(y1))
    y1_grad_np = np.array(X.grad.data)

    X.grad.zero_()

    y2 = S2(X)
    y2_np = np.array(y2.detach().data)
    y2.backward(torch.ones_like(y2))
    y2_grad_np = np.array(X.grad.data)

    X.grad.zero_()

    y3 = S3(X.cuda())
    y3_np = np.array(y2.detach().cpu().data)
    y3.backward(torch.ones_like(y3))
    y3_grad_np = np.array(X.grad.detach().cpu().data)

    np.testing.assert_array_almost_equal(y1_np, y2_np, decimal=4)
    np.testing.assert_array_almost_equal(y1_grad_np, y2_grad_np, decimal=4)
    np.testing.assert_array_almost_equal(y1_np, y3_np, decimal=4)
    np.testing.assert_array_almost_equal(y1_grad_np, y3_grad_np, decimal=4)
    print('Implementations give the same results!')


if __name__ == '__main__':
    test_equal()
