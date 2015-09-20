import pyopencl as cl
import kernels
import numpy as np
import time
from scipy.linalg import solve_banded

def scipy_solve_banded(a, b, c, rhs):
    '''
    Solve the tridiagonal system described
    by a, b, c, and rhs.
    a: lower off-diagonal array (first element ignored)
    b: diagonal array
    c: upper off-diagonal array (last element ignored)
    rhs: right hand side of the system
    '''
    l_and_u = (1, 1)
    ab = np.vstack([np.append(0, c[:-1]),
                    b,
                    np.append(a[1:], 0)])
    x = solve_banded(l_and_u, ab, rhs)
    return x

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

platform = cl.get_platforms()[0]
if 'NVIDIA' in platform.name:
    device = platform.get_devices()[rank%2]
else:
    device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
prg = kernels.get_kernels(ctx)

nx = 16
ny = 16
nz = 16

a, b, c, k1, k2, b_first, k1_first, k1_last = \
        _precompute_coefficients(nx, [1, 2, 1./4, 1, 1./4, 2, 1])
d = np.random.rand(nz, ny, nx)
d_copy = d.copy()

a_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
b_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
c_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
k1_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
k2_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
b_first_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
k1_first_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
k1_last_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(np.log2(nx)*8))
d_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, int(nx*ny*nz*8))

cl.enqueue_copy(queue, a_g, a)
cl.enqueue_copy(queue, b_g, b)
cl.enqueue_copy(queue, c_g, c)
cl.enqueue_copy(queue, k1_g, k1)
cl.enqueue_copy(queue, k2_g, k2)
cl.enqueue_copy(queue, b_first_g, b_first)
cl.enqueue_copy(queue, k1_first_g, k1_first)
cl.enqueue_copy(queue, k1_last_g, k1_last)
cl.enqueue_copy(queue, d_g, d)

bx = nx
by = 2
bz = 2
prg.PrecomputedCR(queue,
        [nx, ny, nz], [bx, by, bz], a_g, b_g, c_g, d_g, k1_g, k2_g,
            b_first_g, k1_first_g, k1_last_g, np.int32(nx), np.int32(ny),
                np.int32(nz), np.int32(bx), np.int32(by),
                    cl.LocalMemory(int(np.log2(nx)*8)), cl.LocalMemory(int(np.log2(nx)*8)),
                        cl.LocalMemory(int(np.log2(nx)*8)), cl.LocalMemory(int(bz*by*bx*8)),
                            cl.LocalMemory(int(np.log2(nx)*8)), cl.LocalMemory(int(np.log2(nx)*8)), 
                                cl.LocalMemory(int(np.log2(nx)*8)), cl.LocalMemory(int(np.log2(nx)*8)),
                                    cl.LocalMemory(int(np.log2(nx)*8)))

evt = cl.enqueue_copy(queue, d, d_g)
evt.wait()
print d[4, 5, :]

a = np.ones(nx)*(1./4)
b = np.ones(nx)*(1.0)
c = np.ones(nx)*(1./4)
a[-1] = 2.0
c[0] = 2.0

x = scipy_solve_banded(a, b, c, d_copy[4, 5, :])
print x
