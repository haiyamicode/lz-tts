from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "ctc_forced_aligner.ctc_forced_aligner",
            ["ctc_forced_aligner/forced_align_impl.cpp"],
            extra_compile_args=["-O3"],
        )
    ]
)
