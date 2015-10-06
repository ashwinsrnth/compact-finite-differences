import sys
sys.path.append('../../../code/cuda')
import numpy as np
from mpi4py import MPI
from gpuDA import *
from compact import CompactFiniteDifferenceSolver
import sys
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

args = sys.argv

nz, ny, nx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
npz, npy, npx = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

da = DA(comm, (nz, ny, nx), (npz, npy, npx), 1)
x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
f = x*np.cos(x*y) + np.sin(z)*y
dfdx_true = -(x*y)*np.sin(x*y) + np.cos(x*y)

dx = x[0, 0, 1] - x[0, 0, 0]

cfd = CompactFiniteDifferenceSolver(da)

start = cuda.Event()
end = cuda.Event()

for i in range(10):
    f_d = gpuarray.to_gpu(f)
    start.record()
    dfdx_d = cfd.dfdx(f_d, dx)
    end.record()
    comm.Barrier()
    end.synchronize()
    
    if rank == 0:
        print start.time_till(end)*1e-3

MPI.Finalize()
