import numpy as np
import kernels

class PThomas:
    def __init__(self, shape):
        '''
        Create context for pThomas (thread-parallel Thomas algorithm)
        '''
        self.nz, self.ny, self.nx = shape 
        self.pThomas, = kernels.get_funcs('kernels.cu', 'pThomasKernel')
    
    def solve(self, a_d, b_d, c_d, c2_d, x_d):
        self.pThomas.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intc])
        self.pThomas.prepared_call((self.nz*self.ny, 1, 1), (self.nx, 1, 1),
             a_d.gpudata, b_d.gpudata, c_d.gpudata, c2_d.gpudata, x_d.gpudata, np.int32(self.nx))
