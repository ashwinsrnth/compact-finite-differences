import pyopencl as cl
from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

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

def get_3d_function_and_derivs(x, y, z):
    f = z*y*np.sin(x) + z*x*np.sin(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.sin(y) + y*np.sin(z)
    dfdy = z*np.sin(x) + z*x*np.cos(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.sin(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz


# solving for the x-derivative in parallel
# using a compact finite difference scheme

comm = MPI.COMM_WORLD
size = comm.Get_size()

NZ, NY, NX = 72, 72, 72
npz, npy, npx = int(size**(1./3)), int(size**(1./3)), int(size**(1./3))
nz, ny, nx = NZ/npz, NY/npy, NX/npx
dx, dy, dz = 2*np.pi/(NX-1), 2*np.pi/(NY-1), 2*np.pi/(NZ-1)

comm = comm.Create_cart([npz, npy, npx])
rank = comm.Get_rank()
size = comm.Get_size()
mz, my, mx = comm.Get_topo()[2]

#------------------------------------------------------------------------------
# initialize the function for the entire domain at
# rank 0, and scatter it:

if rank == 0:
    x, y, z = np.meshgrid(np.linspace(0, 2*np.pi, NX), np.linspace(0, 2*np.pi, NY),
                    np.linspace(0, 2*np.pi, NZ), indexing='ij')
    x, y, z = x.transpose().copy(), y.transpose().copy(), z.transpose().copy()
    f, dfdx_true, _, _ = get_3d_function_and_derivs(x, y, z)
else:
    f, dfdx_true, _, _ = None, None, None, None

dfdx_true_local = np.zeros([nz, ny, nx], dtype=np.float64)
f_local = np.zeros([nz, ny, nx], dtype=np.float64)

start_z, start_y, start_x = mz*nz, my*ny, mx*nx
subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                    [nz, ny, nx], [start_z, start_y, start_x])
subarray = subarray_aux.Create_resized(0, 8)
subarray.Commit()

start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

comm.Scatterv([f, np.ones(size, dtype=np.int), displs, subarray],
    [f_local, MPI.DOUBLE], root=0)

comm.Scatterv([dfdx_true, np.ones(size, dtype=np.int), displs, subarray],
    [dfdx_true_local, MPI.DOUBLE], root=0)


# prepare the RHS at rank = 0 and then scatter it:

if rank == 0:
    d = np.zeros([NZ, NY, NX], dtype=np.float64)
    d[:, :, 1:-1] = (3./4)*(f[:, :, 2:] - f[:, :, :-2])/dx
    d[:, :, 0] = (1./(2*dx))*(-5*f[:,:, 0] + 4*f[:, :, 1] + f[:, :, 2])
    d[:, :, -1] = -(1./(2*dx))*(-5*f[:, :, -1] + 4*f[:, :, -2] + f[:, :, -3])
else:
    d = None

# do a quick test on the RHS
# if rank == 0:
#     a_test = np.ones([NX], dtype=np.float64)*(1./4)
#     b_test = np.ones([NX], dtype=np.float64)*(1.)
#     c_test = np.ones([NX], dtype=np.float64)*(1./4)
#     a_test[-1] = 2.
#     c_test[0] = 2.
#     x_test = scipy_solve_banded(a_test, b_test, c_test, d[NZ/2,NY/2,:])
#     plt.plot(f[NZ/2, NY/2, :])
#     plt.plot(x_test)
#     plt.savefig('test.png')

d_local = np.zeros([nz, ny, nx], dtype=np.float64)

start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

comm.Scatterv([d, np.ones(size, dtype=np.int), displs, subarray],
    [d_local, MPI.DOUBLE], root=0)

#------------------------------------------------------------------------------
# create the LHS for the tridiagonal system of the compact difference scheme:
a_line_local = np.ones(nx, dtype=np.float64)*(1./4)
b_line_local = np.ones(nx, dtype=np.float64)
c_line_local = np.ones(nx, dtype=np.float64)*(1./4)

if mx == 0:
    c_line_local[0] = 2.0
    a_line_local[0] = 0.0

if mx == npx-1:
    a_line_local[-1] = 2.0
    c_line_local[-1] = 0.0

#------------------------------------------------------------------------------

# each processor computes x_R, x_LH_line and x_UH_line:
r_LH_line = np.zeros(nx, dtype=np.float64)
r_UH_line = np.zeros(nx, dtype=np.float64)
r_LH_line[-1] = -c_line_local[-1]
r_UH_line[0] = -a_line_local[0]

x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

x_R = np.zeros([nz, ny, nx], dtype=np.float64)
for i in range(nz):
    for j in range(ny):
        x_R[i,j,:] = scipy_solve_banded(a_line_local, b_line_local, c_line_local, d_local[i,j,:])
comm.Barrier()

#------------------------------------------------------------------------------
# the first and last elements in x_LH and x_UH,
# and also the first and last "faces" in x_R,
# need to be gathered at rank 0:

if rank == 0:
    x_LH_global = np.zeros([2*size], dtype=np.float64)
    x_UH_global = np.zeros([2*size], dtype=np.float64)
else:
    x_LH_global = None
    x_UH_global = None

lengths = np.ones(size)*2
displacements = np.arange(0, 2*size, 2)

comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
    [x_LH_global, lengths, displacements, MPI.DOUBLE])

comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
    [x_UH_global, lengths, displacements, MPI.DOUBLE])

# # checking for correct gather:
# for i in range(27):
#     if rank==i:
#         print x_UH_line[0], x_U H_line[-1]
#     comm.Barrier()
#
# if rank == 0:
#     print x_UH_global

#------------------------------------------------------------------------------
if rank == 0:
    x_R_global = np.zeros([nz, ny, 2*size], dtype=np.float64)
else:
    x_R_global = None

start_z, start_y, start_x = 0, 0, 2*rank
subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, 2*size],
                    [nz, ny, 2], [start_z, start_y, start_x])
subarray = subarray_aux.Create_resized(0, 8)
subarray.Commit()

start_index = np.array(2*rank, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

x_R_faces = np.zeros([nz, ny, 2], dtype=np.float64)
x_R_faces[:, :, 0] = x_R[:, :, 0].copy()
x_R_faces[:, :, 1] = x_R[:, :, -1].copy()

comm.Barrier()
comm.Gatherv([x_R_faces, MPI.DOUBLE],
    [x_R_global, np.ones(size, dtype=np.int), displs, subarray], root=0)

# checking for correct gather:
# if rank == 13:
#     print x_R[:,:,0]
#     print x_R[:,:,-1]
# comm.Barrier()
# if rank == 0:
#     print x_R_global[:,:,26:28]

#------------------------------------------------------------------------------
# Assemble the matrix to compute the transfer parameters

if rank == 0:
    a_reduced = np.zeros([2*size], dtype=np.float64)
    b_reduced = np.zeros([2*size], dtype=np.float64)
    c_reduced = np.zeros([2*size], dtype=np.float64)
    d_reduced = np.zeros([nz, ny, 2*size], dtype=np.float64)
    d_reduced[...] = x_R_global

    a_reduced[0::2] = -1.
    a_reduced[1::2] = x_UH_global[1::2]
    b_reduced[0::2] = x_UH_global[0::2]
    b_reduced[1::2] = x_LH_global[1::2]
    c_reduced[0::2] = x_LH_global[0::2]
    c_reduced[1::2] = -1.

    a_reduced[0::2*npx], c_reduced[0::2*npx], d_reduced[:,:,0::2*npx] = 0.0, 0.0, 0.0
    b_reduced[0::2*npx] = 1.0
    a_reduced[-1::-2*npx], c_reduced[-1::-2*npx], d_reduced[:,:,-1::-2*npx] = 0.0, 0.0, 0.0
    b_reduced[-1::-2*npx] = 1.0

    a_reduced[1::2*npx] = 0.
    c_reduced[-2::-2*npx] = 0.

    params = np.zeros([nz, ny, 2*size])
    for i in range(nz):
        for j in range(ny):
            params[i, j, :] = scipy_solve_banded(a_reduced, b_reduced, c_reduced, -d_reduced[i, j, :])

    np.testing.assert_allclose(params[:, :, 0::2*npx], 0)
    np.testing.assert_allclose(params[:, :, -1::-2*npx], 0)

else:
    params = None

comm.Barrier()

#------------------------------------------------------------------------------
# Scatter the parameters back

params_local = np.zeros([nz, ny, 2], dtype=np.float64)

start_index = np.array(2*rank, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

comm.Scatterv([params, np.ones(size, dtype=int), displs, subarray],
    [params_local, MPI.DOUBLE])

# Test the scatter:
# if rank == 2:
#     print params_local
# comm.Barrier()
# if rank == 0:
#     print params[:,:,4:6]

alpha = params_local[:,:,0]
beta = params_local[:,:,1]

# note the broadcasting below!
comm.Barrier()
dfdx_local = x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
comm.Barrier()

# Gather solution:
start_z, start_y, start_x = mz*nz, my*ny, mx*nx
subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                    [nz, ny, nx], [start_z, start_y, start_x])
subarray = subarray_aux.Create_resized(0, 8)
subarray.Commit()

start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

if rank == 0:
    dfdx = np.zeros([NZ, NY, NX], dtype=np.float64)
else:
    dfdx = None

comm.Gatherv([dfdx_local, MPI.DOUBLE],
    [dfdx, np.ones(size, dtype=np.int), displs, subarray], root=0)

if rank == 0:
    print np.mean(abs(dfdx-dfdx_true)/np.mean(abs(dfdx))), dx
    err = abs(dfdx-dfdx_true)
    plt.plot(f[NZ/2, NY/2, :], '-')
    plt.plot(dfdx[NZ/2, NY/2, :], '--')
    plt.savefig('fun.png')
    plt.close()

    plt.plot(err[nz/2, ny/2, :], 'o')
    plt.savefig('errline.png')
