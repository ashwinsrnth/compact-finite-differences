import pyopencl as cl
import pyopencl.array as cl_array
import kernels
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import *
import pthomas
import sharedcyclicreduction
import kernels
import mpiDA

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

def test_RHS_dfdx():
    nz = 4
    ny = 8
    nx = 16
    f = np.random.rand(nz+2, ny+2, nx+2)
    dx = 1.0
    r_true = np.zeros([nz, ny, nx])
    r_true[:, :, :] = (3./(4*dx))*(f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2])
    r_true[:, :, 0] = (1./(2*dx))*(
            -5*f[1:-1, 1:-1, 1] +
            4*f[1:-1, 1:-1, 2] +
            f[1:-1, 1:-1, 3])
    r_true[:, :, -1] = -(1./(2*dx))*(
            -5*f[1:-1, 1:-1, -2] +
            4*f[1:-1, 1:-1, -3] +
            f[1:-1, 1:-1, -4])
    r = np.zeros_like(r_true, dtype=np.float64)
    f_d = cl_array.to_device(queue, f)
    r_d = cl_array.to_device(queue, r)
    compute_RHS_dfdx, = kernels.get_funcs(context, 'kernels.cl',
            'computeRHSdfdx')
    compute_RHS_dfdx(queue, [nx, ny, nz], None,
            f_d.data, r_d.data, np.float64(dx),
                np.int32(nx), np.int32(ny), np.int32(nz),
                    np.int32(0), np.int32(1))
    r = r_d.get()
    assert_allclose(r, r_true)

def test_RHS_dfdy():
    nz = 4
    ny = 16
    nx = 8
    f = np.random.rand(nz+2, ny+2, nx+2)
    dy = 1.0
    r_true = np.zeros([nz, ny, nx])
    r_true[:, :, :] = (3./(4*dy))*(f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1])
    r_true[:, 0, :] = (1./(2*dy))*(
            -5*f[1:-1, 1, 1:-1] +
            4*f[1:-1, 2, 1:-1] +
            f[1:-1, 3, 1:-1])
    r_true[:, -1, :] = -(1./(2*dy))*(
            -5*f[1:-1, -2, 1:-1] +
            4*f[1:-1, -3, 1:-1] +
            f[1:-1, -4, 1:-1])
    r = np.zeros_like(r_true, dtype=np.float64)
    f_d = cl_array.to_device(queue, f)
    r_d = cl_array.to_device(queue, r)
    compute_RHS_dfdy, = kernels.get_funcs(context, 'kernels.cl',
            'computeRHSdfdy')
    compute_RHS_dfdy(queue, [nx, ny, nz], None,
            f_d.data, r_d.data, np.float64(dy),
                np.int32(nx), np.int32(ny), np.int32(nz),
                    np.int32(0), np.int32(1))
    r = r_d.get()
    assert_allclose(r, r_true)

def test_copy_faces():
    nz = 3
    ny = 4
    nx = 5
    a = np.random.rand(nz, ny, nx)
    a_faces = np.empty([nz, ny, 2], dtype=np.float64)

    a_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, nz*ny*nx*8)
    a_faces_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, nz*ny*2*8)

    cl.enqueue_copy(queue, a_g, a)
    
    copy_faces, = kernels.get_funcs(context, 'kernels.cl', 'copyFaces')
    copy_faces(queue,
            [1, ny, nz], None, 
                a_g, a_faces_g, np.int32(nx), np.int32(ny), np.int32(nz))
    cl.enqueue_copy(queue, a_faces, a_faces_g)

    a_faces_true = a[:,:,[0,-1]].copy()
    assert_allclose(a_faces, a_faces_true)

def test_pThomas():
    nz = 3
    ny = 4
    nx = 5

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()

    solver = pthomas.PThomas(context, queue, (nz, ny, nx), a, b, c)
    d_d = cl_array.to_device(queue, d)
    evt = solver.solve(d_d.data)
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i,j,:])

def test_sum_solutions():
    nz = 3
    ny = 4
    nx = 5

    x_R = np.random.rand(nz, ny, nx)
    x_UH = np.random.rand(nx)
    x_LH = np.random.rand(nx)
    alpha = np.random.rand(nz, ny)
    beta = np.random.rand(nz, ny)

    x_R_d = cl_array.to_device(queue, x_R)
    x_UH_d = cl_array.to_device(queue, x_UH)
    x_LH_d = cl_array.to_device(queue, x_LH)
    alpha_d = cl_array.to_device(queue, alpha)
    beta_d = cl_array.to_device(queue, beta)
        
    sum_solutions, = kernels.get_funcs(context, 'kernels.cl', 'sumSolutionsdfdx3D')

    sum_solutions(queue,
            [nx, ny, nz], None,
                x_R_d.data, x_UH_d.data, x_LH_d.data,
                    alpha_d.data, beta_d.data,
                        np.int32(nx), np.int32(ny), np.int32(nz))

    x_R_calc = x_R_d.get()
    x_R_true = (x_R + np.einsum('ij,k->ijk', alpha, x_UH) +
                np.einsum('ij,k->ijk', beta, x_LH))
    assert_allclose(x_R_calc, x_R_true)

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
