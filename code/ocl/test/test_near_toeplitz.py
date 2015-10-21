import pyopencl as cl
import pyopencl.array as cl_array
from scipy.linalg import solve_banded
from numpy.testing import *

import sys
sys.path.append('..')
from near_toeplitz import *

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
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

solver = NearToeplitzSolver(context, queue, (1, 1, 32),
        (1., 2., 3., 4., 5, 6., 7.))

d = np.random.rand(1, 1, 32)
d_d = cl_array.to_device(queue, d)
solver.solve(d_d, (1, 1))
x = d_d.get()

a = np.ones(32, dtype=np.float64)*(3.)
b = np.ones(32, dtype=np.float64)*(4.)
c = np.ones(32, dtype=np.float64)*(5.)
b[0] = 1.
c[0] = 2.
a[-1] = 6
b[-1] = 7.
x_true = scipy_solve_banded(a, b, c, d.ravel())

assert_allclose(x.ravel(), x_true.ravel())
