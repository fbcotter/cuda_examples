from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='soft_thresh_cuda',
    ext_modules=[
        CUDAExtension('soft_thresh_cuda', [
            'soft_thresh_cuda.cpp',
            'soft_thresh.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
