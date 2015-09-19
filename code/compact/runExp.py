from compactFD import CompactFiniteDifferenceSolver
import numpy as np
from numpy.testing import *
from mpi4py import MPI
import pyopencl as cl
import matplotlib.pyplot as plt
import sys
import socket

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
    npz, npy, npx = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
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

    (NZ, NY, NX)= prob_size

    nx = NX/npx
    ny = NY/npy
    nz = NZ/npz

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

    comm.Barrier()

    f_g = cl.Buffer(ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                (nz+2)*(ny+2)*(nx+2)*8)
    x_g = cl.Buffer(ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                nz*ny*nx*8)

    (f_local, event) = cl.enqueue_map_buffer(queue, f_g,
        cl.map_flags.WRITE | cl.map_flags.READ, 0,
        (nz+2, ny+2, nx+2), np.float64)
    (dfdx_global, event) = cl.enqueue_map_buffer(queue, x_g,
            cl.map_flags.WRITE | cl.map_flags.READ, 0,
            (nz, ny, nx), np.float64)

    #f_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, (nz+2)*(ny+2)*(nx+2)*8)
    #x_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nz*ny*nx*8)
    #f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
    #dfdx_global = np.zeros([nz, ny, nx], dtype=np.float64)

 
    cfd = CompactFiniteDifferenceSolver(ctx, queue, comm, (NZ, NY, NX))

    for i in range(20):
        cfd.dfdx(f_global, dx, dfdx_global, f_local, f_g, x_g, print_timings=True)
        comm.Barrier()

    if rank == 0: print np.mean(abs(dfdx_global - dfdx_true_global)/np.mean(abs(dfdx_true_global)))
    comm.Barrier()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    NZ = int(sys.argv[1])
    NY = int(sys.argv[2])
    NX = int(sys.argv[3])
    comm.Barrier()
    print 'Rank: ', rank, 'Hostname: ', socket.gethostname()
    comm.Barrier()
    run((NZ, NY, NX))
