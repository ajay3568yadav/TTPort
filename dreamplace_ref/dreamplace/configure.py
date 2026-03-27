"""Minimal DREAMPlace configure stub for TTPort.

The compile_configurations dict mirrors what the full DREAMPlace CMake build
produces. Setting CUDA_FOUND to "FALSE" forces every DREAMPlace op to take its
CPU-only code path; the CUDA kernels are therefore not required at runtime.

To enable CUDA paths, build DREAMPlace with CMake and point PYTHONPATH at the
real build output instead of this dreamplace_ref/ tree.
"""

compile_configurations: dict = {
    "CUDA_FOUND": "FALSE",
    "TORCH_VERSION": "0",
    "PYTHON_VERSION": "3",
}
