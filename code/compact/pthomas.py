import pyopencl as cl
import numpy as np
import kernels

class PThomas:
    def __init__(self, ctx, queue, shape, a, b, c):
        '''
        Create context for pThomas (thread-parallel Thomas algorithm)
        '''
        self.ctx = ctx
        self.queue = queue
        self.platforms = self.ctx.devices[0].platform
        self.nz, self.ny, self.nx = shape 
    
        mf = cl.mem_flags

        self.a_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)
        self.b_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)
        self.c_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)
        self.c2_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)

        cl.enqueue_copy(queue, self.a_g, a)
        cl.enqueue_copy(queue, self.b_g, b)
        cl.enqueue_copy(queue, self.c_g, c)
        cl.enqueue_copy(queue, self.c2_g, c)
        
        self.pThomas, = kernels.get_funcs(ctx, 'kernels.cl', 'pThomasKernel')
    
    def solve(self, x_g):
        evt = self.pThomas(self.queue, [self.nz*self.ny], None,
             self.a_g, self.b_g, self.c_g, x_g, self.c2_g, np.int32(self.nx))
        return evt 
