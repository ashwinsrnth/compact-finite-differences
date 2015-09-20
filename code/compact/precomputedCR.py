import pyopencl as cl
import kernels
import numpy as np


class PrecomputedCR:

    def __init__(self, ctx, queue, shape, coeffs):
        self.ctx = ctx
        self.queue = queue
        self.nz, self.ny, self.nx = shape
        self.coeffs = coeffs

        self.prg = kernels.get_kernels(self.ctx)

        # allocate memory for solver:
        self.a_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.b_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.c_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k2_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.b_first_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_first_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_last_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(np.log2(self.nx))*8)

        # compute coefficients a, b, etc.,
        a, b, c, k1, k2, b_first, k1_first, k1_last = _precompute_coefficients(self.nx, self.coeffs)

        # copy coefficients to buffers:
        cl.enqueue_copy(self.queue, self.a_g, a)
        cl.enqueue_copy(self.queue, self.b_g, b)
        cl.enqueue_copy(self.queue, self.c_g, c)
        cl.enqueue_copy(self.queue, self.k1_g, k1)
        cl.enqueue_copy(self.queue, self.k2_g, k2)
        cl.enqueue_copy(self.queue, self.b_first_g, b_first)
        cl.enqueue_copy(self.queue, self.k1_first_g, k1_first)
        cl.enqueue_copy(self.queue, self.k1_last_g, k1_last)

    def solve(self, x_g, blocks):
        '''
        Solve the system in blocks of size 'blocks',
        specified as [bz, by]
        '''
        nz, ny, nx = self.nz, self.ny, self.nx
        bz, by = blocks
        s1 = int(np.log2(nx)*8)
        s2 = int(bz*by*nx*8)
        
        evt = self.prg.PrecomputedCR(self.queue,
                [nx, ny, nz], [nx, by, bz],
                self.a_g, self.b_g, self.c_g, x_g,
                self.k1_g, self.k2_g,
                self.b_first_g, self.k1_first_g, self.k1_last_g,
                np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(nx), np.int32(by),
                np.float64(self.coeffs[0]), np.float64(self.coeffs[1]),
                np.float64(self.coeffs[2]), np.float64(self.coeffs[3]), np.float64(self.coeffs[4]),
                cl.LocalMemory(s1), cl.LocalMemory(s1), cl.LocalMemory(s1),
                cl.LocalMemory(s2),
                cl.LocalMemory(s1), cl.LocalMemory(s1), cl.LocalMemory(s1),
                cl.LocalMemory(s1), cl.LocalMemory(s1))
        evt.wait()

def _precompute_coefficients(nx, coeffs):
    '''
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of constant coefficients,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(nx)-1,
    as opposed to arrays sized nx.

    Values at the first and last point at each step
    need to be stored seperately.

    The last values for a and b are required only at
    the final stage of forward reduction (the 2-by-2 solve),
    so for convenience, these two scalar values are stored
    at the end of arrays a and b.

    -- See the paper
    "Fast Tridiagonal Solvers on the GPU"
    '''
    # these arrays technically have length 1 more than required:
    log2_nx = int(np.log2(nx))

    a = np.zeros(log2_nx, np.float64)
    b = np.zeros(log2_nx, np.float64)
    c = np.zeros(log2_nx, np.float64)
    k1 = np.zeros(log2_nx, np.float64)
    k2 = np.zeros(log2_nx, np.float64)

    b_first = np.zeros(log2_nx, np.float64)
    k1_first = np.zeros(log2_nx, np.float64)
    k1_last = np.zeros(log2_nx, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_nx - 1
    for i in range(num_reductions):
        if i == 0:
            k1[i] = ai/bi
            k2[i] = ci/bi
            a[i] = -ai*k1[i]
            b[i] = bi - ci*k1[i] - ai*k2[i]
            c[i] = -ci*k2[i]

            k1_first[i] = ai/b1
            b_first[i] = bi - c1*k1_first[i] - ai*k2[i]

            k1_last[i] = an/bi
            a_last = -(ai)*k1_last[i]
            b_last = bn - (ci)*k1_last[i]
        else:
            k1[i] = a[i-1]/b[i-1]
            k2[i] = c[i-1]/b[i-1]
            a[i] = -a[i-1]*k1[i]
            b[i] = b[i-1] - c[i-1]*k1[i] - a[i-1]*k2[i]
            c[i] = -c[i-1]*k2[i]

            k1_first[i] = a[i-1]/b_first[i-1]
            b_first[i] = b[i-1] - c[i-1]*k1_first[i] - a[i-1]*k2[i]

            k1_last[i] = a_last/b[i-1]
            a_last = -a[i-1]*k1_last[i]
            b_last = b_last - c[i-1]*k1_last[i]

    # put the last values for a and b at the end of the arrays:
    a[-1] = a_last
    b[-1] = b_last

    return a, b, c, k1, k2, b_first, k1_first, k1_last


