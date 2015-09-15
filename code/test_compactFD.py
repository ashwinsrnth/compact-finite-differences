from compactFD import CompactFiniteDifferenceSolver
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import pyopencl as cl
import matplotlib.pyplot as plt
import tools

def get_3d_function_and_derivs_1(x, y, z):
    f = z*y*np.sin(x) + z*x*np.sin(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.sin(y) + y*np.sin(z)
    dfdy = z*np.sin(x) + z*x*np.cos(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.sin(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz

def get_3d_function_and_derivs_2(x, y, z):
    f = z*y*np.sin(x) + z*x*np.cos(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.cos(y) + y*np.sin(z)
    dfdy = z*np.sin(x) - z*x*np.sin(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.cos(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz

def get_3d_function_and_derivs_3(x, y, z):
    f = np.sin(y)
    dfdx = 0.
    dfdy = np.cos(y)
    dfdz = 0.
    return f, dfdx, dfdy, dfdz

def rel_err(a, b, method='max'):
    if method == 'mean':
        return np.mean(abs(a-b)/np.max(abs(b)))
    else:
        return np.max(abs(a-b)/np.max(abs(b)))

def test_compactFD_dfdx():
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

    NX = NY = NZ = 128

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

    f_global, dfdx_true_global, _, _ = get_3d_function_and_derivs_1(x_global, y_global, z_global)

    cfd = CompactFiniteDifferenceSolver(ctx, queue, comm, (NZ, NY, NX))
    dfdx_global = np.zeros_like(f_global, dtype=np.float64)
    cfd.dfdx(f_global, dx, dfdx_global)

    print rel_err(dfdx_global, dfdx_true_global), rel_err(dfdx_global, dfdx_true_global, method='mean')

    if rank == 0:
        dfdx = np.zeros([NZ, NY, NX], dtype=np.float64)
    else:
        dfdx = None

    tools.gather_3D(comm, dfdx_global, dfdx)

    if rank == 0:
        plt.plot(dfdx[NZ/2, NY/2, :])
        plt.savefig('temp.png')


if __name__ == "__main__":
    test_compactFD_dfdx()
