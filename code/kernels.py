import pyopencl as cl

def get_kernels(ctx):
    platform = ctx.devices[0].platform
    with open('kernels.cl') as f:
        kernel_text = f.read()
        if 'NVIDIA' in platform.name:
            kernel_text = '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n' + kernel_text
            prg = cl.Program(ctx, kernel_text).build(options=['-cl-nv-arch sm_35'])
        else:
            prg = cl.Program(ctx, kernel_text).build(options=['-O2'])
    return prg
