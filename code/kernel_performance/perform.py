import sys
sys.path.append('..')
import kernels

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

def _precompute_coefficients(system_size, coeffs):
    ''' 
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of constant coefficients,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(system_size)-1,
    as opposed to arrays sized system_size.

    Values at the first and last point at each step
    need to be stored seperatel+y.

    The last values for a and b are required only at
    the final stage of forward reduction (the 2-by-2 solve),
    so for convenience, these two scalar values are stored
    at the end of arrays a and b.

    -- See the paper
    "Fast Tridiagonal Solvers on the GPU"
    '''
    # these arrays technically have length 1 more than required:
    log2_system_size = int(np.log2(system_size))

    a = np.zeros(log2_system_size, np.float64)
    b = np.zeros(log2_system_size, np.float64)
    c = np.zeros(log2_system_size, np.float64)
    k1 = np.zeros(log2_system_size, np.float64)
    k2 = np.zeros(log2_system_size, np.float64)

    b_first = np.zeros(log2_system_size, np.float64)
    k1_first = np.zeros(log2_system_size, np.float64)
    k1_last = np.zeros(log2_system_size, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_system_size - 1
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

if __name__ == "__main__":

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    (computeRHS, globalForwardReduction_x, globalBackSubstitution_x,
        globalForwardReduction_y, globalBackSubstitution_y,
            globalForwardReduction_z, globalBackSubstitution_z) =  kernels.get_funcs(ctx,
                    '../kernels.cl', 'computeRHS', 'globalForwardReduction_x',
                         'globalBackSubstitution_x', 'globalForwardReduction_y',
                            'globalBackSubstitution_y', 'globalForwardReduction_z',
                                'globalBackSubstitution_z')

    N = 32
    nz, ny, nx = N, N, N
    dz, dy, dx = 2*np.pi/(nz-1), 2*np.pi/(ny-1), 2*np.pi/(nx-1)
    coeffs = (1., 2., 1./4, 1., 1./4, 2., 1.)
    a, b, c, k1, k2, b_first, k1_first, k1_last = _precompute_coefficients(nx, coeffs)
    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs
    
    a_d = cl_array.to_device(queue, a)
    b_d = cl_array.to_device(queue, b)
    c_d = cl_array.to_device(queue, c)
    k1_d = cl_array.to_device(queue, k1)
    k2_d = cl_array.to_device(queue, k2)
    b_first_d = cl_array.to_device(queue, b_first)
    k1_first_d = cl_array.to_device(queue, k1_first)
    k1_last_d = cl_array.to_device(queue, k1_last)

    z, y, x = np.meshgrid(
    np.linspace(0, (nz-1)*dz, nz),
    np.linspace(0, (ny-1)*dy, ny),
    np.linspace(0, (nx-1)*dx, nx),
    indexing='ij')

    f = np.sin(x) + np.cos(y) + 2*(z**3)

    dfdx_true = np.cos(x)
    dfdy_true = -np.sin(y)
    dfdz_true = 6*z**2

    f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
    f_local[1:-1, 1:-1, 1:-1] = f

    f_local_d = cl_array.to_device(queue, f_local)
    dx_d = cl.array.Array(queue, [nz, ny, nx], dtype=np.float64)
    dy_d = cl.array.Array(queue, [nz, ny, nx], dtype=np.float64)
    dz_d = cl.array.Array(queue, [nz, ny, nx], dtype=np.float64)

    computeRHS(queue, [nz, ny, nx], None,
                    f_local_d.data, dx_d.data, dy_d.data, dz_d.data, np.float64(dx), np.float64(dy), np.float64(dz),
                       np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(0), np.int32(1))

    
    t1 = time.time()
    # CR algorithm
    # ============================================
    stride = 1
    for i in np.arange(int(np.log2(nx))):
        stride *= 2
        evt = globalForwardReduction_x(queue, [nx/stride, ny, nz], [nx/stride, 1, 1],
            a_d.data, b_d.data, c_d.data, dx_d.data, k1_d.data, k2_d.data,
                b_first_d.data, k1_first_d.data, k1_last_d.data,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                        np.int32(stride))
        evt.wait()

    # `stride` is now equal to `nx`
    for i in np.arange(int(np.log2(nx))-1):
        stride /= 2
        evt = globalBackSubstitution_x(queue, [nx/stride, ny, nz], [nx/stride, 1, 1],
            a_d.data, b_d.data, c_d.data, dx_d.data, b_first_d.data,
                np.float64(b1), np.float64(c1),
                    np.float64(ai), np.float64(bi), np.float64(ci),
                        np.int32(nx), np.int32(ny), np.int32(nz),
                            np.int32(stride))
        evt.wait()
    t2 = time.time()

    print 'x-derivative: ', t2-t1 

    t1 = time.time()
    # CR algorithm
    # ============================================
    stride = nx
    for i in np.arange(int(np.log2(ny))):
        stride *= 2
        evt = globalForwardReduction_y(queue, [nx, nx*ny/stride, nz], [1, nx*ny/stride, 1],
            a_d.data, b_d.data, c_d.data, dy_d.data, k1_d.data, k2_d.data,
                b_first_d.data, k1_first_d.data, k1_last_d.data,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                        np.int32(stride))
        evt.wait()
        
    # # `stride` is now equal to `nx*ny`
    for i in np.arange(int(np.log2(ny))-1):
        stride /= 2
        evt = globalBackSubstitution_y(queue, [nx, nx*ny/stride, nz], [1, nx*ny/stride, 1],
            a_d.data, b_d.data, c_d.data, dy_d.data, b_first_d.data,
                np.float64(b1), np.float64(c1),
                    np.float64(ai), np.float64(bi), np.float64(ci),
                        np.int32(nx), np.int32(ny), np.int32(nz),
                            np.int32(stride))
        evt.wait()
    t2 = time.time()

    print 'y-derivative: ', t2-t1        

    t1 = time.time()
    # CR algorithm
    # ============================================
    stride = nx*ny
    for i in np.arange(int(np.log2(nz))):
        stride *= 2
        evt = globalForwardReduction_z(queue, [nx, ny, nx*ny*nz/stride], [1, 1, nx*ny*nz/stride],
            a_d.data, b_d.data, c_d.data, dz_d.data, k1_d.data, k2_d.data,
                b_first_d.data, k1_first_d.data, k1_last_d.data,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                        np.int32(stride))
        evt.wait()
        
    # # `stride` is now equal to `nx*ny`
    for i in np.arange(int(np.log2(nz))-1):
        stride /= 2
        evt = globalBackSubstitution_z(queue, [nx, ny, nx*ny*nz/stride], [1, 1, nx*ny*nz/stride],
            a_d.data, b_d.data, c_d.data, dz_d.data, b_first_d.data,
                np.float64(b1), np.float64(c1),
                    np.float64(ai), np.float64(bi), np.float64(ci),
                        np.int32(nx), np.int32(ny), np.int32(nz),
                            np.int32(stride))
        evt.wait()
    t2 = time.time()

    print 'z-derivative: ', t2-t1





