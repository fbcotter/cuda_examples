from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='soft_thresh',
    ext_modules=[
        CppExtension('soft_thresh', ['soft_thresh.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
