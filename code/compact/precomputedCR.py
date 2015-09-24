import pyopencl as cl
import numpy as np
import kernels

'''
A tridiagonal solver for solving
the tridiagonal system that arises in the evaulation
of derivatives using compact finite-difference schemes.

For example, our solver handles the tridiagonal system
for the following tridiagonal scheme:

alpha(f'[i-1] - f'[i+1]) + f'[i] = a(f[i+1] - f[i-1])/dx

With alpha=1/4, a=3/4

Along with the following implicit equation at the boundaries:

f'[1] + 2f'[2] =  (-5f[1] + 4f[2] + f[3])/2dx

The tridiagonal system is then of the form:

1       2       .       .       .       .
1/ 4    1       1/4     .       .       .
.       1/4     1       1/4     .       .
.       .       1/4     1       1/4     .
.       .       .       1/4     1       1/4
.       .       .       .       2       1
'''

class PrecomputedCR:

    def __init__(self, ctx, queue, shape, coeffs):
        '''
        Create context for the Cyclic Reduction Solver
        that solves a tridiagonal system with
        diagonals
        a = (1/4, 1/4, 1/4 .... 2)
        b[:] = (1, 1, 1, 1... 1)
        c[:] = (2, 1/4, 1/4, ... 1/4)

        Parameters
        ----------
        ctx: PyOpenCL context
        queue: PyOpenCL command queue
        system_size: The size of the tridiagonal system.
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            [b1, c1, ai, bi, ci, an, bn]
        '''
        self.ctx = ctx
        self.queue = queue
        self.device = self.ctx.devices[0]
        self.platform = self.device.platform
        self.nz, self.ny, self.nx = shape
        self.coeffs = coeffs

        mf = cl.mem_flags

        # check that system_size is a power of 2:
        assert np.int(np.log2(self.nx)) == np.log2(self.nx)

        # allocate memory for solver:
        self.a_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.b_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.c_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k2_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.b_first_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_first_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)
        self.k1_last_g = cl.Buffer(self.ctx, mf.READ_WRITE, size=int(np.log2(self.nx))*8)

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

        self.forward_reduction, self.back_substitution = kernels.get_funcs(self.ctx, 'kernels.cl', 'globalForwardReduction', 'globalBackSubstitution')

    def solve(self, x_g, blocks, print_profile=False):
        '''
            Solve the tridiagonal system
            for rhs d, given storage for the solution
            vector in x.
            Additionally, OpenCL corresponding
            OpenCL buffers d_g and x_g must be provided.
        '''
        [b1, c1,
            ai, bi, ci,
                an, bn] = self.coeffs

        bz, by = blocks

        # CR algorithm
        # ============================================
        stride = 1
        for i in np.arange(int(np.log2(self.nx))):
            stride *= 2
            evt = self.forward_reduction(self.queue, [self.nx/stride, self.ny, self.nz], [self.nx/stride, by, bz],
                self.a_g, self.b_g, self.c_g, x_g, self.k1_g, self.k2_g,
                    self.b_first_g, self.k1_first_g, self.k1_last_g,
                        np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                            np.int32(stride))
            evt.wait()
        
        # `stride` is now equal to `nx`
        for i in np.arange(int(np.log2(self.nx))-1):
            stride /= 2
            evt = self.back_substitution(self.queue, [self.nx/stride, self.ny, self.nz], [self.nx/stride, by, bz],
                self.a_g, self.b_g, self.c_g, x_g, self.b_first_g,
                    np.float64(b1), np.float64(c1),
                        np.float64(ai), np.float64(bi), np.float64(ci),
                            np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                                np.int32(stride))
            evt.wait()
        # ============================================
        
def _precompute_coefficients(system_size, coeffs):
    '''
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of constant coefficients,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(system_size)-1,
    as opposed to arrays sized system_size.

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
    log2_system_size = int(np.log2(system_size))

    a = np.zeros(log2_system_size, np.float64)
    b = np.zeros(log2_system_size, np.float64)
    c = np.zeros(log2_system_size, np.float64)
    k1 = np.zeros(log2_system_size, np.float64)
    k2 = np.zeros(log2_system_size, np.float64)

    b_first = np.zeros(log2_system_size, np.float64)
    k1_first = np.zeros(log2_system_size, np.float64)
    k1_last = np.zeros(log2_system_size, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_system_size - 1
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
