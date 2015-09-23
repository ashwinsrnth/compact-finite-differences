import pyopencl as cl
import pyopencl.array as cl_array
import kernels
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import *

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
prg = kernels.get_kernels(context)

def test_copy_faces():
    nz = 3
    ny = 4
    nx = 5
    a = np.random.rand(nz, ny, nx)
    a_faces = np.empty([nz, ny, 2], dtype=np.float64)

    a_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, nz*ny*nx*8)
    a_faces_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, nz*ny*2*8)

    cl.enqueue_copy(queue, a_g, a)

    prg.copyFaces(queue,
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

    a_d = cl_array.to_device(queue, a)
    b_d = cl_array.to_device(queue, b)
    c_d = cl_array.to_device(queue, c)
    c2_d = cl_array.to_device(queue, c)
    d_d = cl_array.to_device(queue, d)

    prg.pThomasKernel(queue,
            [nz*ny], None,
                a_d.data, b_d.data, c_d.data, d_d.data, c2_d.data, np.int32(nx))
    
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

    prg.sumSolutionsdfdx3D(queue,
            [nx, ny, nz], None,
                x_R_d.data, x_UH_d.data, x_LH_d.data,
                    alpha_d.data, beta_d.data,
                        np.int32(nx), np.int32(ny), np.int32(nz))

    x_R_calc = x_R_d.get()
    x_R_true = (x_R + np.einsum('ij,k->ijk', alpha, x_UH) +
                np.einsum('ij,k->ijk', beta, x_LH))
    assert_allclose(x_R_calc, x_R_true)

def test_single_line_cyclic_reduction():
    nz = 3
    ny = 4
    nx = 8

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()

    a_d = cl_array.to_device(queue, a)
    b_d = cl_array.to_device(queue, b)
    c_d = cl_array.to_device(queue, c)
    d_d = cl_array.to_device(queue, d)

    prg.singleLineCyclicReduction(queue,
            [nx, ny, nz], [nx, 1, 1],
                a_d.data, b_d.data, c_d.data, d_d.data,
                    np.int32(nx), np.int32(ny), np.int32(nz), np.int32(nx),
                        cl.LocalMemory(nx*8), cl.LocalMemory(nx*8),
                            cl.LocalMemory(nx*8), cl.LocalMemory(nx*8))
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i,j,:])
            
def test_multi_line_cyclic_reduction():
    nz = 16
    ny = 16
    nx = 32

    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    d = np.random.rand(nz, ny, nx)
    d_copy = d.copy()

    a_d = cl_array.to_device(queue, a)
    b_d = cl_array.to_device(queue, b)
    c_d = cl_array.to_device(queue, c)
    d_d = cl_array.to_device(queue, d)

    by = 2
    bz = 2
    prg.multiLineCyclicReduction(queue,
            [nx, ny, nz], [nx, by, bz],
                a_d.data, b_d.data, c_d.data, d_d.data,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                        np.int32(nx), np.int32(by),
                            cl.LocalMemory(nx*8), cl.LocalMemory(nx*8),
                            cl.LocalMemory(nx*8), cl.LocalMemory(nx*by*bz*8))
    d = d_d.get()

    for i in range(nz):
        for j in range(ny):
            x_true = scipy_solve_banded(a, b, c, d_copy[i,j,:])
            assert_allclose(x_true, d[i,j,:])
