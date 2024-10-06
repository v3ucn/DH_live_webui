from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("webui_dh_live.py")
)