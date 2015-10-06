import numpy as np
import kernels

class ReducedSolver:
    def __init__(self, shape):
        '''
        Create context for pThomas (thread-parallel Thomas algorithm)
        '''
        self.nz, self.ny, self.nx = shape 
        self.solver, = kernels.get_funcs('kernels.cu', 'reducedSolverKernel')
    
    def solve(self, a_d, b_d, c_d, c2_d, x_d):
        self.solver.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intc, np.intc, np.intc])
        self.solver.prepared_call((self.nx*self.ny, 1, 1), (1, 1, 1),
             a_d.gpudata, b_d.gpudata, c_d.gpudata, c2_d.gpudata, x_d.gpudata,
                np.int32(self.nx), np.int32(self.ny), np.int32(self.nz))
