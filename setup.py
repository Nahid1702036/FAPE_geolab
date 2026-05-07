"""
setup.py  —  fape_geolab
Drives the CUDAExtension build; invoked automatically by pip / build.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    "-O3", "--use_fast_math",
    "-gencode=arch=compute_70,code=sm_70",   # Volta
    "-gencode=arch=compute_80,code=sm_80",   # Ampere A100
    "-gencode=arch=compute_86,code=sm_86",   # Ampere RTX 3000
    "-gencode=arch=compute_89,code=sm_89",   # Ada RTX 4000
    "-gencode=arch=compute_90,code=sm_90",   # Hopper H100
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-DCUDA_HAS_FP16=1",
]

setup(
    ext_modules=[
        CUDAExtension(
            name="fape_geolab._C",
            sources=[
                "csrc/fape_pytorch_bridge.cpp",
                "csrc/cuda/fape_surgical_kernel.cu",
            ],
            include_dirs=[os.path.join(ROOT, "csrc")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
