import pyopencl as cl
import kernels
import numpy as np
import time
from scipy.linalg import solve_banded
from numpy.testing import *

import precomputedCR

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
    device = platform.get_devices()[rank%2]
else:
    device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

nx = 16
ny = 16
nz = 16

solver = precomputedCR.PrecomputedCR(ctx, queue, [nz, ny, nx], [1., 2., 1./4, 1., 1./4, 2., 1.])

d_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nx*ny*nz*8)
d = np.random.rand(nz, ny, nx)
cl.enqueue_copy(queue, d_g, d)
d_copy = d.copy()

solver.solve(d_g, [2, 2])

evt = cl.enqueue_copy(queue, d, d_g)
evt.wait()

a = np.ones(nx)*(1./4)
b = np.ones(nx)*(1.0)
c = np.ones(nx)*(1./4)
a[-1] = 2.0
c[0] = 2.0

x = np.zeros_like(d, dtype=np.float64)

for i in range(nz):
    for j in range(ny):
        x[i, j, :] = scipy_solve_banded(a, b, c, d_copy[i, j, :])
assert_allclose(x, d)
