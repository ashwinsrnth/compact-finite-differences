import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sys
sys.path.append('../../../code')
from near_toeplitz import *
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose
import time

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

args = sys.argv
nx = int(args[1])
ny = int(args[2])
nz = int(args[3])

# get a context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# generate the coefficient arrays and right-hand-side
d = np.ones([nz, ny, nx], dtype=np.float64)
d_d = cl_array.to_device(queue, d)

# transfer to device
cfd = NearToeplitzSolver(ctx, queue, [nz, ny, nx], [1., 2., 1./4, 1., 1./4, 2., 1.])

for i in range(10):
    t1 = time.time()
    cfd.solve(d_d, [1, 1])
    t2 = time.time()
    print t2-t1

'''
a = np.ones(nx, dtype=np.float64)*(1./4)
b = np.ones(nx, dtype=np.float64)*(1.)
c = np.ones(nx, dtype=np.float64)*(1./4)
a[-1] = 2.
c[0] = 2.

x = d_d.get()

for i in range(nz):
    for j in range(ny):
        x_true = scipy_solve_banded(a, b, c, d[i, j, :])
        assert_allclose(x_true, x[i,j,:])
'''
