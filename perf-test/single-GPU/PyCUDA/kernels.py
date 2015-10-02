from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.compiler as compiler

def get_funcs(file_name, *args):
    with open(file_name) as f:
        kernel_source = f.read()
    module = compiler.SourceModule(kernel_source, options=['-O2'], arch='sm_35')
    
    funcs = []
    for func_name in args:
        funcs.append(module.get_function(func_name))
    return funcs
