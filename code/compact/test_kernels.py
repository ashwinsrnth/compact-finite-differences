import pyopencl as cl
import pyopencl.array as cl_array
import kernels
import numpy as np
from numpy.testing import *
import kernels

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
