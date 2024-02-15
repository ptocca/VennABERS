from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = '0.0.1'


ext_modules = [
    Pybind11Extension(
        "libVennABERS",
        ["lib/libVennABERS.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name='VennABERS',
    version=__version__,
    description='Venn-ABERS predictor',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    py_modules=['VennABERS'],
    install_requires=["scikit-learn>=1.1.0"]
)