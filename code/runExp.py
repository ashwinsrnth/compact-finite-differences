import compactFD
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import matplotlib.pyplot as plt
import sys

def get_3d_function_and_derivs_1(x, y, z):
    f = z*y*np.sin(x) + z*x*np.sin(y) + x*y*np.sin(z)
    dfdx = z*y*np.cos(x) + z*np.sin(y) + y*np.sin(z)
    dfdy = z*np.sin(x) + z*x*np.cos(y) + x*np.sin(z)
    dfdz = y*np.sin(x) + x*np.sin(y) + x*y*np.cos(z)
    return f, dfdx, dfdy, dfdz

def run(prob_size):

    comm = MPI.COMM_WORLD
    args = sys.argv

    comm.Barrier()
    t1 = MPI.Wtime()
     
    size = comm.Get_size()
    npz, npy, npx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert npz*npy*npx == size

    comm = comm.Create_cart([npz, npy, npx])
    rank = comm.Get_rank()

    NX = NY = NZ = prob_size

    nx = NX/npz
    ny = NX/npy
    nz = NX/npx

    mz, my, mx = comm.Get_topo()[2]

    dx = 2*np.pi/(NX-1)
    dy = 2*np.pi/(NY-1)
    dz = 2*np.pi/(NZ-1)
    
    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Initialization: ', t2-t1

    comm.Barrier()
    t1 = MPI.Wtime()

    x_start, y_start, z_start = mx*nx*dx, my*ny*dy, mz*nz*dz
    x_local, y_local, z_local = np.meshgrid(
        np.linspace(x_start, x_start + (nx-1)*dx, nx),
        np.linspace(y_start, y_start + (ny-1)*dy, ny),
        np.linspace(z_start, z_start + (nz-1)*dz, nz),
        indexing='ij')

    x_local, y_local, z_local = x_local.transpose().copy(), y_local.transpose().copy(), z_local.transpose().copy()
    f_local, dfdx_true_local, _, _ = get_3d_function_and_derivs_1(x_local, y_local, z_local)

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Computing function and true derivative: ', t2-t1

    dfdx_local = compactFD.dfdx(comm, f_local, dx)

    print np.mean(abs(dfdx_local - dfdx_true_local)/np.mean(abs(dfdx_true_local)))
    comm.Barrier()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for prob_size in 24, 48, 96, 192, 286:
        if rank == 0:
            print prob_size
        run(prob_size)
        if rank == 0:
            print '--------------'
