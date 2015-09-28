import pyopencl as cl
import numpy as np
import kernels

class PThomas:
    def __init__(self, ctx, queue, shape):
        '''
        Create context for pThomas (thread-parallel Thomas algorithm)
        '''
        self.ctx = ctx
        self.queue = queue
        self.platforms = self.ctx.devices[0].platform
        self.nz, self.ny, self.nx = shape 
        self.pThomas, = kernels.get_funcs(ctx, 'kernels.cl', 'pThomasKernel')
    
    def solve(self, a_g, b_g, c_g, c2_g, x_g):
        evt = self.pThomas(self.queue, [self.nz*self.ny], None,
             a_g, b_g, c_g, c2_g, x_g, np.int32(self.nx))
        return evt 
