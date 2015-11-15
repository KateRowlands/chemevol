from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="integrals", ext_modules=cythonize('integrals.pyx'),include_dirs=[numpy.get_include()], extra_compile_args = ["-ffast-math"])
