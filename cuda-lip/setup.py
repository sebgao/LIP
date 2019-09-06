from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='LIP',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('lip_cuda_interface', [
            'src/lip_cuda.cpp',
            'src/lip_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
