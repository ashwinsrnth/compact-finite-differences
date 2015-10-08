import sys
sys.path.append('../../../code/cuda')
import numpy as np
from mpi4py import MPI
from gpuDA import *
from compact import CompactFiniteDifferenceSolver
import sys
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda_init

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

f_d = gpuarray.to_gpu(f)
f_local_d = da.create_local_vector()
x_d = da.create_global_vector()

print f_d.shape

for i in range(10):
    t1 = MPI.Wtime()
    cfd.dfdx(f_d, dx, x_d, f_local_d)
    cuda.Context.synchronize()
    comm.Barrier()
    t2 = MPI.Wtime()
    if rank == 0: print 'Total: ', t2-t1
