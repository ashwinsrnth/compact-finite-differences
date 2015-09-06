import compactFD
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import matplotlib.pyplot as plt

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

def test_compactFD_dfdx():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size==27
    comm = comm.Create_cart([3, 3, 3])
    rank = comm.Get_rank()
    N = 36

    if rank == 0:
        x_range = np.linspace(0, 2*np.pi, N)
        x, y, z = np.meshgrid(x_range, x_range, x_range, indexing='ij')
        x, y, z = x.transpose().copy(), y.transpose().copy(), z.transpose().copy()
        f, dfdx_true, _, _ = get_3d_function_and_derivs_2(x, y, z)
    else:
        f, dfdx_true = None, None

    dx = 2*np.pi/(N-1)
    dfdx = compactFD.dfdx(comm, f, dx)

    if rank == 0:
        plt.plot(dfdx_true[N/2, N/2, :], linewidth=4, alpha=0.5, label='true')
        plt.plot(dfdx[N/2, N/2, :], '-', linewidth=2, label='computed')
        plt.legend()
        plt.savefig('dfdx.png')
        plt.close()
        print np.mean(abs(dfdx-dfdx_true)/abs(np.max(dfdx_true)))
        print 'Plot of solution at z=N/2, y=N/2 saved to file.'

def test_compactFD_dfdy():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size==27
    comm = comm.Create_cart([3, 3, 3])
    rank = comm.Get_rank()
    N = 36

    if rank == 0:
        x_range = np.linspace(0, 2*np.pi, N)
        x, y, z = np.meshgrid(x_range, x_range, x_range, indexing='ij')
        x, y, z = x.transpose().copy(), y.transpose().copy(), z.transpose().copy()
        f, dfdx_true, dfdy_true, _ = get_3d_function_and_derivs_2(x, y, z)
    else:
        f, dfdx_true, dfdy_true = None, None, None

    dy = 2*np.pi/(N-1)
    dfdy = compactFD.dfdy(comm, f, dy)

    if rank == 0:
        plt.plot(dfdy_true[N/2, :, N/2], linewidth=4, alpha=0.5, label='true')
        plt.plot(dfdy[N/2, :, N/2], '-', linewidth=2, label='computed')
        plt.legend()
        plt.savefig('dfdy.png')
        plt.close()
        print np.mean(abs(dfdy-dfdy_true)/abs(np.max(dfdy_true)))
        print 'Plot of solution at x=N/2, z=N/2 saved to file.'

def test_compactFD_dfdz():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size==27
    comm = comm.Create_cart([3, 3, 3])
    rank = comm.Get_rank()
    N = 36

    if rank == 0:
        x_range = np.linspace(0, 2*np.pi, N)
        x, y, z = np.meshgrid(x_range, x_range, x_range, indexing='ij')
        x, y, z = x.transpose().copy(), y.transpose().copy(), z.transpose().copy()
        f, dfdx_true, dfdy_true, dfdz_true = get_3d_function_and_derivs_2(x, y, z)
    else:
        f, dfdx_true, dfdy_true, dfdz_true = None, None, None, None

    dz = 2*np.pi/(N-1)
    dfdz = compactFD.dfdz(comm, f, dz)

    if rank == 0:
        plt.plot(dfdz_true[:, N/2, N/2], linewidth=4, alpha=0.5, label='true')
        plt.plot(dfdz[:, N/2, N/2], '-', linewidth=2, label='computed')
        plt.legend()
        plt.savefig('dfdz.png')
        plt.close()
        print np.mean(abs(dfdz-dfdz_true)/abs(np.max(dfdz_true)))
        print 'Plot of solution at x=N/2, y=N/2 saved to file.'

if __name__ == "__main__":
    test_compactFD_dfdx()
    test_compactFD_dfdy()
    test_compactFD_dfdz()
