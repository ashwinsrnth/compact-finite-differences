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

