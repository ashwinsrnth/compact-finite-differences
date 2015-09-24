import pyopencl as cl
import numpy as np
import sandbox


class SharedMemCyclicReduction:

    def __init__(self, ctx, queue, shape, a, b, c):
        self.ctx = ctx
        self.queue = queue
        self.platforms = self.ctx.devices[0].platform
        self.nz, self.ny, self.nx = shape 
    
        mf = cl.mem_flags

        self.a_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)
        self.b_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)
        self.c_g = cl.Buffer(ctx, mf.READ_WRITE, self.nx*8)

        cl.enqueue_copy(queue, self.a_g, a)
        cl.enqueue_copy(queue, self.b_g, b)
        cl.enqueue_copy(queue, self.c_g, c)
        
        self.solver, = sandbox.get_funcs(ctx, 'kernels.cl',
                'multiLineCyclicReduction')

    def solve(self, x_g, by, bz):
        evt = self.solver(self.queue, [self.nx, self.ny, self.nz], [self.nx, by, bz],
                self.a_g, self.b_g, self.c_g, x_g,
                    np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                        np.int32(self.nx), np.int32(by),
                            cl.LocalMemory(self.nx*8),
                            cl.LocalMemory(self.nx*8),
                            cl.LocalMemory(self.nx*8),
                            cl.LocalMemory(self.nx*by*bz*8))
        return evt


