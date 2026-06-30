from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        "src.starling.utils.monotonic_align.core",
        [str(Path("src/starling/utils/monotonic_align/core.pyx"))],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
