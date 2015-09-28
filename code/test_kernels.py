import sys
sys.path.append('..')
import pyopencl as cl
import pyopencl.array as cl_array
import kernels
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import *
import pthomas
import sharedcyclicreduction
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

def test_pThomas():
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

def test_single_line_cyclic_reduction():
            
    nz = 16
    ny = 16
    nx = 32

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()
    d_d = cl_array.to_device(queue, d)

    by = 1
    bz = 1
    
    solver = sharedcyclicreduction.SharedMemCyclicReduction(
            context, queue, (nz, ny, nx), a, b, c)
    solver.solve(d_d.data, by, bz)
    
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i, j, :])

def test_multi_line_cyclic_reduction():
    '''
    This test is failing because every
    thread in shared memory is writing to the same location,
    leading to overwrites in the case of a_l, b_l and c_l.


    '''
    
    nz = 16
    ny = 16
    nx = 32

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()
    d_d = cl_array.to_device(queue, d)

    by = 2
    bz = 2
    
    solver = sharedcyclicreduction.SharedMemCyclicReduction(
            context, queue, (nz, ny, nx), a, b, c)
    solver.solve(d_d.data, by, bz)
    
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i, j, :])

def test_precomputed_cyclic_reduction():
   pass 
