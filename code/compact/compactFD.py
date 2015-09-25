import mpiDA
import kernels
from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array as cl_array
import near_toeplitz
import pThomas
from mpi_util import *

def get_3d_function_and_derivs_1(x, y, z):
    f = z*y*np.sin(x) + z*x*np.sin(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.sin(y) + y*np.sin(z)
    dfdy = z*np.sin(x) + z*x*np.cos(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.sin(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz

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

comm = MPI.COMM_WORLD
size = comm.Get_size()
size_per_dir = int(size**(1./3))
comm = comm.Create_cart([size_per_dir, size_per_dir, size_per_dir])
rank = comm.Get_rank()
platform = cl.get_platforms()[0]
if 'NVIDIA' in platform.name:
    device = platform.get_devices()[rank%2]
else:
    device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

NX = 32
NY = 64
NZ = 64

nx = NX/size_per_dir
ny = NY/size_per_dir
nz = NZ/size_per_dir

npz, npy, npx = comm.Get_topo()[0]
mz, my, mx = comm.Get_topo()[2]

dx = 2*np.pi/(NX-1)
dy = 2*np.pi/(NY-1)
dz = 2*np.pi/(NZ-1)

x_start, y_start, z_start = mx*nx*dx, my*ny*dy, mz*nz*dz
z_global, y_global, x_global = np.meshgrid(
    np.linspace(z_start, z_start + (nz-1)*dz, nz),
    np.linspace(y_start, y_start + (ny-1)*dy, ny),
    np.linspace(x_start, x_start + (nx-1)*dx, nx),
    indexing='ij')

f_global, dfdx_true_global, dfdy_true_global, _ = get_3d_function_and_derivs_1(x_global, y_global, z_global)

# preprocessing - get the kernels
copy_faces, compute_RHS_dfdx,  sum_solutions= kernels.get_funcs(ctx, 'kernels.cl',
    'copyFaces', 'computeRHSdfdx', 'sumSolutionsdfdx3D')

# preprocessing - compute the RHS
da = mpiDA.DA(comm, [nz, ny, nx], [npz, npy, npx], 1)
f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
d = np.zeros([nz, ny, nx], dtype=np.float64)
da.global_to_local(f_global, f_local)
f_d = cl_array.to_device(queue, f_local)
d_d = cl_array.to_device(queue, d)
compute_RHS_dfdx(queue, [nz, ny, nx], None,
        f_d.data, d_d.data,
            np.float64(dx), np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(mx), np.int32(npx))

# preprocessing -get the line_comm
line_comm = MPI_get_line(comm, 0)

# preprocessing - solve for x_LH and x_UH:
a = np.ones(nx, dtype=np.float64)*(1./4)
b = np.ones(nx, dtype=np.float64)
c = np.ones(nx, dtype=np.float64)*(1./4)
r_UH = np.zeros(nx, dtype=np.float64)
r_LH = np.zeros(nx, dtype=np.float64)

if mx == 0:
    c[0] =  2.0
    a[0] = 0.0

if mx == npx-1:
    a[-1] = 2.0
    c[-1] = 0.0

r_UH[0] = -a[0]
r_LH[-1] = -c[-1]

x_UH = scipy_solve_banded(a, b, c, r_UH)
x_LH = scipy_solve_banded(a, b, c, r_LH)

# preprocessing - setup the reduced system:
x_UH_line = np.zeros(2*npx, dtype=np.float64)
x_LH_line = np.zeros(2*npx, dtype=np.float64)
lengths = np.ones(npx)*2
displacements = np.arange(0, 2*npx, 2)

line_comm.Gatherv([np.array([x_UH[0], x_UH[-1]]), MPI.DOUBLE],
   [x_UH_line, lengths, displacements, MPI.DOUBLE])

line_comm.Gatherv([np.array([x_LH[0], x_LH[-1]]), MPI.DOUBLE],
   [x_LH_line, lengths, displacements, MPI.DOUBLE])

if mx == 0:
    a_reduced = np.zeros(2*npx, dtype=np.float64)
    b_reduced = np.zeros(2*npx, dtype=np.float64)
    c_reduced = np.zeros(2*npx, dtype=np.float64)
    a_reduced[0::2] = -1.
    a_reduced[1::2] = x_UH_line[1::2]
    b_reduced[0::2] = x_UH_line[0::2]
    b_reduced[1::2] = x_LH_line[1::2]
    c_reduced[0::2] = x_LH_line[0::2]
    c_reduced[1::2] = -1.
    a_reduced[0], c_reduced[0] = 0.0, 0.0
    b_reduced[0] = 1.0
    a_reduced[-1], c_reduced[-1] = 0.0, 0.0
    b_reduced[-1] = 1.0
    a_reduced[1] = 0.
    c_reduced[-2] = 0.
    reduced_solver = pThomas.pThomas(ctx, queue, [nz, ny, 2*npx],
            a_reduced, b_reduced, c_reduced)

# solve the local systems for x_R:
coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
if mx == 0:
    coeffs[1] = 2.
if mx == npx-1:
    coeffs[-2] = 2.
block_solver = near_toeplitz.NearToeplitzSolver(ctx, queue, (nz, ny, nx),
    coeffs)
block_solver.solve(d_d.data, [2, 2])

# copy the faces:
x_R_faces = np.zeros([nz, ny, 2])
x_R_faces_line = np.zeros([nz, ny, 2*npx])

x_R_faces_d = cl_array.to_device(queue, x_R_faces)
copy_faces(queue, [1, ny, nz], None,
        d_d.data, x_R_faces_d.data, np.int32(nx), np.int32(ny), np.int32(nz))
x_R_faces[...] = x_R_faces_d.get()

lengths = np.ones(npx)
displacements = np.arange(0, 2*npx, 2)
subarray = face_type(line_comm, [nz, ny, nx])
line_comm.Gatherv([x_R_faces, MPI.DOUBLE],
        [x_R_faces_line, lengths, displacements, subarray])

# solve the reduced system:
if mx == 0:
    x_R_faces_line[:, :, 0] = 0.0
    x_R_faces_line[:, :, -1] = 0.0
    d_reduced_d = cl_array.to_device(queue, -x_R_faces_line)
    reduced_solver.solve(d_reduced_d.data)
    params = d_reduced_d.get()
else:
    params = None

# scatter parameters
params_local = np.zeros([nz, ny, 2], dtype=np.float64)
line_comm.Scatterv([params, lengths, displacements, subarray],
       [params_local, MPI.DOUBLE]) 

alpha = params_local[:, :, 0].copy()
beta = params_local[:, :, 1].copy()

x_UH_d = cl_array.to_device(queue, x_UH)
x_LH_d = cl_array.to_device(queue, x_LH)
alpha_d = cl_array.to_device(queue, alpha)
beta_d = cl_array.to_device(queue, beta)

sum_solutions(queue, [nx, ny, nz], None,
        d_d.data, x_UH_d.data, x_LH_d.data, alpha_d.data, beta_d.data,
            np.int32(nx), np.int32(ny), np.int32(nz))
x = d_d.get()

if rank == 3:
    plt.subplot(211)
    plt.pcolor(x[:, :, 8])
    plt.subplot(212)
    plt.pcolor(dfdx_true_global[:, :, 8])
    plt.savefig('compare.png')
    plt.close()

# preprocessing - get the kernels
copy_faces, compute_RHS_dfdy,  sum_solutions= kernels.get_funcs(ctx, 'kernels.cl',
    'copyFaces', 'computeRHSdfdy', 'sumSolutionsdfdx3D')

# preprocessing - compute the RHS
da = mpiDA.DA(comm, [nz, ny, nx], [npz, npy, npx], 1)
f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
d = np.zeros([nz, ny, nx], dtype=np.float64)
da.global_to_local(f_global, f_local)

compute_RHS_dfdy(queue, [nx, ny, nz], None,
        f_d.data, d_d.data,
            np.float64(dy), np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(my), np.int32(npy))

# transpose:
d = d_d.get()
d = d.transpose(0, 2, 1).copy()
d_d = cl_array.to_device(queue, d)

# preprocessing -get the line_comm
line_comm = MPI_get_line(comm, 1)

# preprocessing - solve for x_LH and x_UH:
a = np.ones(ny, dtype=np.float64)*(1./4)
b = np.ones(ny, dtype=np.float64)
c = np.ones(ny, dtype=np.float64)*(1./4)
r_UH = np.zeros(ny, dtype=np.float64)
r_LH = np.zeros(ny, dtype=np.float64)

if my == 0:
    c[0] =  2.0
    a[0] = 0.0

if my == npy-1:
    a[-1] = 2.0
    c[-1] = 0.0

r_UH[0] = -a[0]
r_LH[-1] = -c[-1]

x_UH = scipy_solve_banded(a, b, c, r_UH)
x_LH = scipy_solve_banded(a, b, c, r_LH)

# preprocessing - setup the reduced system:
x_UH_line = np.zeros(2*npy, dtype=np.float64)
x_LH_line = np.zeros(2*npy, dtype=np.float64)
lengths = np.ones(npy)*2
displacements = np.arange(0, 2*npy, 2)

line_comm.Gatherv([np.array([x_UH[0], x_UH[-1]]), MPI.DOUBLE],
   [x_UH_line, lengths, displacements, MPI.DOUBLE])

line_comm.Gatherv([np.array([x_LH[0], x_LH[-1]]), MPI.DOUBLE],
   [x_LH_line, lengths, displacements, MPI.DOUBLE])

if my == 0:
    a_reduced = np.zeros(2*npy, dtype=np.float64)
    b_reduced = np.zeros(2*npy, dtype=np.float64)
    c_reduced = np.zeros(2*npy, dtype=np.float64)
    a_reduced[0::2] = -1.
    a_reduced[1::2] = x_UH_line[1::2]
    b_reduced[0::2] = x_UH_line[0::2]
    b_reduced[1::2] = x_LH_line[1::2]
    c_reduced[0::2] = x_LH_line[0::2]
    c_reduced[1::2] = -1.
    a_reduced[0], c_reduced[0] = 0.0, 0.0
    b_reduced[0] = 1.0
    a_reduced[-1], c_reduced[-1] = 0.0, 0.0
    b_reduced[-1] = 1.0
    a_reduced[1] = 0.
    c_reduced[-2] = 0.
    reduced_solver = pThomas.pThomas(ctx, queue, [nz, ny, 2*npy],
            a_reduced, b_reduced, c_reduced)

# solve the local systems for x_R:
coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
if my == 0:
    coeffs[1] = 2.
if my == npy-1:
    coeffs[-2] = 2.
block_solver = near_toeplitz.NearToeplitzSolver(ctx, queue, (nz, nx, ny),
    coeffs)
block_solver.solve(d_d.data, [2, 2])

# copy the faces:
x_R_faces = np.zeros([nz, nx, 2])
x_R_faces_line = np.zeros([nz, nx, 2*npy])

x_R_faces_d = cl_array.to_device(queue, x_R_faces)
copy_faces(queue, [1, nx, nz], None,
        d_d.data, x_R_faces_d.data, np.int32(ny), np.int32(nx), np.int32(nz))
x_R_faces[...] = x_R_faces_d.get()

lengths = np.ones(npy)
displacements = np.arange(0, 2*npy, 2)
subarray = face_type(line_comm, [nz, nx, ny])
line_comm.Gatherv([x_R_faces, MPI.DOUBLE],
        [x_R_faces_line, lengths, displacements, subarray])

# solve the reduced system:
if my == 0:
    x_R_faces_line[:, :, 0] = 0.0
    x_R_faces_line[:, :, -1] = 0.0
    d_reduced_d = cl_array.to_device(queue, -x_R_faces_line)
    reduced_solver.solve(d_reduced_d.data)
    params = d_reduced_d.get()
else:
    params = None

# scatter parameters
params_local = np.zeros([nz, nx, 2], dtype=np.float64)
line_comm.Scatterv([params, lengths, displacements, subarray],
       [params_local, MPI.DOUBLE]) 

alpha = params_local[:, :, 0].copy()
beta = params_local[:, :, 1].copy()

x_UH_d = cl_array.to_device(queue, x_UH)
x_LH_d = cl_array.to_device(queue, x_LH)
alpha_d = cl_array.to_device(queue, alpha)
beta_d = cl_array.to_device(queue, beta)

sum_solutions(queue, [ny, nx, nz], None,
        d_d.data, x_UH_d.data, x_LH_d.data, alpha_d.data, beta_d.data,
            np.int32(ny), np.int32(nx), np.int32(nz))
x = d_d.get()

# transpose it!
x = x.transpose(0, 2, 1).copy()

print np.mean(abs(x - dfdy_true_global)/np.max(abs(dfdy_true_global)))

if rank == 0:
    plt.subplot(211)
    plt.pcolor(x[:, :, 12])
    plt.subplot(212)
    plt.pcolor(dfdy_true_global[:, :, 12])
    plt.savefig('compare.png')
    plt.close()
