import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension


ext_name = 'spmm_cpp'
cpp_src = 'spmm.cpp' 
if torch.version.cuda.startswith('10'):
    cpp_src = 'spmm_original.cpp' 


setup(name=ext_name, 
        ext_modules=[cpp_extension.CppExtension(ext_name, [cpp_src])], 
        cmdclass={'build_ext': cpp_extension.BuildExtension})
