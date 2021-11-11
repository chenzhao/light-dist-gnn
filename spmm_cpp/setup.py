from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='spmm_cpp', 
        ext_modules=[cpp_extension.CppExtension('spmm_cpp', ['cusparse_spmm.cpp'])], 
        cmdclass={'build_ext': cpp_extension.BuildExtension})
