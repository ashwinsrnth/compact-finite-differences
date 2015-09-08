import compactFD
import numpy as np
from numpy.testing import *
from mpi4py import MPI
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

    NX = NY = NZ = 36

    nx = NX/size_per_dir
    ny = NX/size_per_dir
    nz = NX/size_per_dir

    npx, npy, npz = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]

    dx = 2*np.pi/(NX-1)
    dy = 2*np.pi/(NY-1)
    dz = 2*np.pi/(NZ-1)

    x_start, y_start, z_start = mx*nx*dx, my*ny*dy, mz*nz*dz
    x_local, y_local, z_local = np.meshgrid(
        np.linspace(x_start, x_start + (nx-1)*dx, nx),
        np.linspace(y_start, y_start + (ny-1)*dy, ny),
        np.linspace(z_start, z_start + (nz-1)*dz, nz),
        indexing='ij')

    x_local, y_local, z_local = x_local.transpose().copy(), y_local.transpose().copy(), z_local.transpose().copy()
    f_local, dfdx_true_local, _, _ = get_3d_function_and_derivs_1(x_local, y_local, z_local)
    dfdx_local = compactFD.dfdx(comm, f_local, dx)

    print rel_err(dfdx_local, dfdx_true_local), rel_err(dfdx_local, dfdx_true_local, method='mean')

def test_compactFD_dfdy():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    size_per_dir = int(size**(1./3))
    comm = comm.Create_cart([size_per_dir, size_per_dir, size_per_dir])
    rank = comm.Get_rank()

    NX = NY = NZ = 36

    nx = NX/size_per_dir
    ny = NX/size_per_dir
    nz = NX/size_per_dir

    npx, npy, npz = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]

    dx = 2*np.pi/(NX-1)
    dy = 2*np.pi/(NY-1)
    dz = 2*np.pi/(NZ-1)

    x_start, y_start, z_start = mx*nx*dx, my*ny*dy, mz*nz*dz
    x_local, y_local, z_local = np.meshgrid(
        np.linspace(x_start, x_start + (nx-1)*dx, nx),
        np.linspace(y_start, y_start + (ny-1)*dy, ny),
        np.linspace(z_start, z_start + (nz-1)*dz, nz),
        indexing='ij')

    x_local, y_local, z_local = x_local.transpose().copy(), y_local.transpose().copy(), z_local.transpose().copy()
    f_local, _, dfdy_true_local, _ = get_3d_function_and_derivs_1(x_local, y_local, z_local)
    dfdy_local = compactFD.dfdy(comm, f_local, dy)


    if rank == 0:
        dfdy = np.zeros([NZ, NY, NX], dtype=np.float64)
    else:
        dfdy = None

    tools.gather_3D(comm, dfdy_local, dfdy)

    if rank == 0:
        plt.plot(dfdy[NZ/2, :, NX/2])
        plt.savefig('temp.png')

    print rel_err(dfdy_local, dfdy_true_local), rel_err(dfdy_local, dfdy_true_local, method='mean')


def test_compactFD_dfdz():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    size_per_dir = int(size**(1./3))
    comm = comm.Create_cart([size_per_dir, size_per_dir, size_per_dir])
    rank = comm.Get_rank()

    NX = NY = NZ = 36

    nx = NX/size_per_dir
    ny = NX/size_per_dir
    nz = NX/size_per_dir

    npx, npy, npz = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]

    dx = 2*np.pi/(NX-1)
    dy = 2*np.pi/(NY-1)
    dz = 2*np.pi/(NZ-1)

    x_start, y_start, z_start = mx*nx*dx, my*ny*dy, mz*nz*dz
    x_local, y_local, z_local = np.meshgrid(
        np.linspace(x_start, x_start + (nx-1)*dx, nx),
        np.linspace(y_start, y_start + (ny-1)*dy, ny),
        np.linspace(z_start, z_start + (nz-1)*dz, nz),
        indexing='ij')

    x_local, y_local, z_local = x_local.transpose().copy(), y_local.transpose().copy(), z_local.transpose().copy()
    f_local, _, _, dfdz_true_local = get_3d_function_and_derivs_1(x_local, y_local, z_local)
    dfdz_local = compactFD.dfdz(comm, f_local, dz)

    print rel_err(dfdz_local, dfdz_true_local), rel_err(dfdz_local, dfdz_true_local, method='mean')

if __name__ == "__main__":
    test_compactFD_dfdx()
    MPI.COMM_WORLD.Barrier()
    test_compactFD_dfdy()
    MPI.COMM_WORLD.Barrier()
    test_compactFD_dfdz()
