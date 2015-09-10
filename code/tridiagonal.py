import kernels
import pyopencl as cl
import numpy as np
import time


class BatchTridiagonalSolver:
    '''
    Solve tridiagonal systems
    with the same
    left hand side and several right hand sides.
    '''
    def __init__(self, ctx, queue, comm):
        self.comm = comm
        self.ctx = ctx
        self.queue = queue
        self.prg = kernels.get_kernels(ctx)

    def solve(self, a, b, c, d, num_systems, system_size):
        t1 = time.time()
        dfdx = np.zeros(num_systems*system_size, dtype=np.float64)
        t2 = time.time()

        print 'Initial allocation: ', t2-t1

        a_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        b_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        c_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        c2_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        d_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)
        dfdx_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)

        t1 = time.time()

        ta = time.time()
        evt1 = cl.enqueue_copy(self.queue, a_g, a)
        evt1.wait()
        evt2 = cl.enqueue_copy(self.queue, b_g, b)
        evt2.wait()
        evt3 = cl.enqueue_copy(self.queue, c_g, c)
        evt3.wait()
        evt5 = cl.enqueue_copy(self.queue, c2_g, c)
        evt5.wait()
        tb = time.time()
        print 'Time for small buffer copies: ', tb-ta

        ta = time.time()
        evt4 = cl.enqueue_copy(self.queue, d_g, d)
        evt4.wait()
        tb = time.time()

        print 'Time for large buffer copies: ', tb-ta

        t2 = time.time()

        print 'Time for buffer copies: ', t2-t1

        t1 = time.time()

        evt = self.prg.compactTDMA(self.queue, [num_systems], None,
            a_g, b_g, c_g, d_g, dfdx_g, c2_g,
                np.int32(system_size))

        evt.wait()

        t2 = time.time()

        print 'Time for solve: ', t2-t1

        evt = cl.enqueue_copy(self.queue, dfdx, dfdx_g)
        evt.wait()

        return dfdx
