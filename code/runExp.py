from compactFD import CompactFiniteDifferenceSolver
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import pyopencl as cl
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

    platform = cl.get_platforms()[0]
    if 'NVIDIA' in platform.name:
        device = platform.get_devices()[rank%2]
    else:
        device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    NX = NY = NZ = prob_size

    nx = NX/npx
    ny = NY/npy
    nz = NZ/npz

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
    z_local, y_local, x_local = np.meshgrid(
        np.linspace(z_start, z_start + (nz-1)*dz, nz),
        np.linspace(y_start, y_start + (ny-1)*dy, ny),
        np.linspace(x_start, x_start + (nx-1)*dx, nx),
        indexing='ij')

    f_local, dfdx_true_local, _, _ = get_3d_function_and_derivs_1(x_local, y_local, z_local)

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Computing function and true derivative: ', t2-t1

    cfd = CompactFiniteDifferenceSolver(ctx, queue, comm, (NZ, NY, NX))
    dfdx_local = np.zeros_like(f_local, dtype=np.float64)
    cfd.dfdx(f_local, dx, dfdx_local)

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
