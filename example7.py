""" This is the same code as example 1, however it uses unified memory
"""
import numpy as np
import cupy
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from collections import namedtuple
from string import Template
from dtcwt.numpy.lowlevel import colfilter as np_colfilter
Stream = namedtuple('Stream', ['ptr'])

CUDA_NUM_THREADS = 1024


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


with open('filters.cu', 'r') as f:
    cuda_source = f.read()


class RowFilter(Function):

    def __init__(self, weight, klow=None, khigh=None):
        super(RowFilter, self).__init__()
        self.weight = weight
        if klow is None:
            klow = -np.floor((weight.shape[0] - 1) / 2)
            khigh = np.ceil((weight.shape[0] - 1) / 2)
        assert abs(klow) == khigh, "can only do odd filters for the moment"
        self.klow = klow
        self.khigh = khigh
        assert abs(klow) == khigh
        self.f = load_kernel('rowfilter', cuda_source)
        self.fbwd = load_kernel('rowfilter_bwd', cuda_source)

    #  @staticmethod
    def forward(ctx, input):
        assert input.dim() == 4 and input.is_cuda and ctx.weight.is_cuda
        n, ch, h, w = input.shape
        ctx.in_shape = (n, ch, h, w)
        pad_end = 0
        output = torch.zeros((n, ch, h, w + pad_end),
                             dtype=torch.float32,
                             requires_grad=input.requires_grad).cuda()

        with torch.cuda.device_of(input):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[output.data_ptr(), input.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w+pad_end), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    #  @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.in_shape
        n, ch, h, w = grad_out.shape
        grad_input = torch.zeros(in_shape, dtype=torch.float32).cuda()

        with torch.cuda.device_of(grad_out):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[grad_input.data_ptr(), grad_out.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input


class ColFilter(Function):
    def __init__(self, weight, klow=None, khigh=None):
        super(ColFilter, self).__init__()
        self.weight = weight
        if klow is None:
            klow = -np.floor((weight.shape[0] - 1) / 2)
            khigh = np.ceil((weight.shape[0] - 1) / 2)
        assert abs(klow) == khigh, "can only do odd filters for the moment"
        self.klow = klow
        self.khigh = khigh
        self.f = load_kernel('colfilter', cuda_source)
        self.fbwd = load_kernel('colfilter_bwd', cuda_source)

    #  @staticmethod
    def forward(ctx, input):
        assert input.dim() == 4 and input.is_cuda and ctx.weight.is_cuda
        n, ch, h, w = input.shape
        ctx.in_shape = (n, ch, h, w)
        pad_end = 0
        output = torch.zeros((n, ch, h + pad_end, w),
                             dtype=torch.float32,
                             requires_grad=input.requires_grad).cuda()

        with torch.cuda.device_of(input):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[output.data_ptr(), input.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(h+pad_end),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    #  @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.in_shape
        n, ch, h, w = grad_out.shape
        grad_input = torch.zeros(in_shape, dtype=torch.float32).cuda()

        with torch.cuda.device_of(grad_out):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[grad_input.data_ptr(), grad_out.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(h),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input


if __name__ == '__main__':
    # It is important to use float32 as most GPUs still work on single precision
    # floating points. These arrays need 64 MiB of memory each.
    #  a = np.random.randn(16, 3, 64, 64).astype('float32')
    w = np.random.randn(1, 11).astype('float32')
    a = np.repeat(np.expand_dims(np.arange(10),axis=0), repeats=5,
                  axis=0).astype('float32')
    a = np.repeat(np.expand_dims(a, axis=0), repeats=3, axis=0)
    a = np.repeat(np.expand_dims(a, axis=0), repeats=2, axis=0)
    w = np.array([1,1,1,1,1]).astype('float32')
    w = np.expand_dims(w, -1)

    # Some useful functions
    ref_colfilter = lambda x, h: np.stack([
        np.stack([np_colfilter(s, h) for s in c], axis=0)
        for c in x], axis=0)
    ref_rowfilter = lambda x, h: np.stack([
        np.stack([np_colfilter(s.T, h).T for s in c], axis=0)
        for c in x], axis=0)

    y1 = ref_colfilter(a, w)
    y2 = ref_rowfilter(a, w)

    w_col = torch.tensor(w, dtype=torch.float32).cuda()
    w_row = torch.tensor(w, dtype=torch.float32).cuda()
    a_t = torch.tensor(a, requires_grad=True)
    a_t_gpu = a_t.cuda()

    mod = ColFilter(w_col)
    y1_t = mod(a_t_gpu)
    gradcheck(mod, (a_t_gpu,), eps=1e-2, atol=1e-3)

    mod = RowFilter(w_row)
    y2_t = mod(a_t_gpu)
    gradcheck(mod, (a_t_gpu,), eps=1e-2, atol=1e-3)

    np.testing.assert_array_almost_equal(
        y1_t.cpu().detach().numpy(), y1, decimal=4)
    np.testing.assert_array_almost_equal(
        y2_t.cpu().detach().numpy(), y2, decimal=4)
