import sys
sys.path.append('..')
import pyopencl as cl
import pyopencl.array as cl_array
import kernels
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import *
import pthomas
import kernels
from mpi_util import *

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

def test_pthomas():
    nz = 3
    ny = 4
    nx = 5

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()

    solver = pthomas.PThomas(context, queue, (nz, ny, nx))
    a_d = cl_array.to_device(queue, a)
    b_d = cl_array.to_device(queue, b)
    c_d = cl_array.to_device(queue, c)
    c2_d = cl_array.to_device(queue, c)
    d_d = cl_array.to_device(queue, d)
    evt = solver.solve(a_d.data, b_d.data, c_d.data, c2_d.data, d_d.data)
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i,j,:])
    print 'pass'

if __name__ == "__main__":
    test_pthomas() 
