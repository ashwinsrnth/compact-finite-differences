import pyopencl as cl
import kernels
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose

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


platform = cl.get_platforms()[0]
if 'NVIDIA' in platform.name:
    device = platform.get_devices()[0]
else:
    device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
prg = kernels.get_kernels(ctx)

nz = 2
ny = 2
nx = 8

a = np.random.rand(nx)
b = np.random.rand(nx)
c = np.random.rand(nx)
d = np.random.rand(nz, ny, nx)
x = np.empty_like(d, dtype=np.float64)

for i in range(nz):
    for j in range(ny):
        x[i, j, :] = scipy_solve_banded(a, b, c, d[i, j, :])

def blockCyclicReduction(a, b, c, d):
    nz, ny, nx = d.shape
    a_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nx*8)
    b_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nx*8)
    c_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nx*8)
    d_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nx*ny*nz*8)
    cl.enqueue_copy(queue, a_g, a)
    cl.enqueue_copy(queue, b_g, b)
    cl.enqueue_copy(queue, c_g, c)
    cl.enqueue_copy(queue, d_g, d)

    prg.blockCyclicReduction(queue, [nx, ny, nz], [nx, 1, 1],
        a_g, b_g, c_g, d_g,
            np.int32(nx), np.int32(ny), np.int32(nz), np.int32(nx),
                cl.LocalMemory(nx*8), cl.LocalMemory(nx*8), cl.LocalMemory(nx*8), cl.LocalMemory(nx*8))

    cl.enqueue_copy(queue, d, d_g)

blockCyclicReduction(a, b, c, d)
assert_allclose(x, d)
