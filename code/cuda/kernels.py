from pycuda import autoinit
import pycuda.compiler as compiler
import numpy as np
import os

def get_funcs(filename, *args):
    src_dir = os.path.dirname(__file__)
    with open(src_dir + '/' + filename) as f:
        kernel_source = f.read()
    module = compiler.SourceModule(kernel_source, options=['-O2'], arch='sm_35')
    
    funcs = []
    for func_name in args:
        funcs.append(module.get_function(func_name))
    return funcs

