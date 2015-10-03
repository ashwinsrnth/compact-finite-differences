from pycuda import autoinit
import pycuda.compiler as compiler
import numpy as np
import pyopencl as cl
import os

def get_funcs(ctx, filename, *args):
    '''
    Build the code in 'src' and get the
    kernels in args therein
    '''
    src_dir = os.path.dirname(__file__)
    with open(src_dir + '/' + filename) as f:
        src = f.read()
    platform = ctx.devices[0].platform
    if 'NVIDIA' in platform.name:
        src = '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n' + src
        prg = cl.Program(ctx, src).build(options=['-cl-nv-arch sm_35'])
    else:
        prg = cl.Program(ctx, src).build(options=['-O2'])
    funcs = []
    for kernel in args:
        funcs.append(getattr(prg, kernel))
    return funcs

def get_funcs_cu(file_name, *args):
    with open(file_name) as f:
        kernel_source = f.read()
    module = compiler.SourceModule(kernel_source, options=['-O2'], arch='sm_35')
    
    funcs = []
    for func_name in args:
        funcs.append(module.get_function(func_name))
    return funcs

