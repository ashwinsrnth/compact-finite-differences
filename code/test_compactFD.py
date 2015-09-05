import compactFD
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import matplotlib.pyplot as plt

def get_3d_function_and_derivs(x, y, z):
    f = z*y*np.sin(x) + z*x*np.sin(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.sin(y) + y*np.sin(z)
    dfdy = z*np.sin(x) + z*x*np.cos(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.sin(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz

def test_compactFD_simple():
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
        f, dfdx_true, _, _ = get_3d_function_and_derivs(x, y, z)
    else:
        f, dfdx_true = None, None

    dx = 2*np.pi/(N-1)
    dfdx = compactFD.dfdx(comm, f, dx)

    if rank == 0:
        plt.plot(dfdx_true[N/2, N/2, :])
        plt.plot(dfdx[N/2, N/2, :], 'o')
        plt.savefig('computed.png')
        plt.close()
        print np.mean(abs(dfdx-dfdx_true)/abs(np.max(dfdx_true)))

if __name__ == "__main__":
    test_compactFD_simple()
