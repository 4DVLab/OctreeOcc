from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='octree_ops',
    ext_modules=[
        CUDAExtension(
            name='octree_ops',
            sources=[
                'ops/octree_ops.cpp',
                'ops/octree_ops_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)