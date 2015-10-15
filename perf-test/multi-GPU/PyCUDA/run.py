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
import socket
import time
from numpy.testing import *

args = sys.argv
nz, ny, nx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
npz, npy, npx = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
solver = sys.argv[7]

assert (solver == 'globalmem' or solver == 'templated')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

da = DA(comm, (nz, ny, nx), (npz, npy, npx), 1)
line_da = da.get_line_DA(0)

x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
f = x*np.cos(x*y) + np.sin(z)*y
dfdx_true = -(x*y)*np.sin(x*y) + np.cos(x*y)

dx = x[0, 0, 1] - x[0, 0, 0]

cfd = CompactFiniteDifferenceSolver(line_da, solver)

f_d = gpuarray.to_gpu(f)
f_local_d = da.create_local_vector()
x_d = da.create_global_vector()

da.comm.Barrier()


if rank == 0: print 'Solving a {0} x {1} x {2} system on {3} x {4} x {5} processors'.format(nz*npz,
        ny*npy, nx*npx, npz, npy, npx)
if rank == 0: print '------------------------------------------------------'

for i in range(20):
    if rank == 0: print 'Run {0}'.format(i+1)
    f_d = gpuarray.to_gpu(f)
    da.comm.Barrier()
    cuda.Context.synchronize()
    t1 = MPI.Wtime()
    cfd.dfdx(f_d, dx, x_d, f_local_d)
    cuda.Context.synchronize()
    da.comm.Barrier()
    t2 = MPI.Wtime()
    if rank == 0: print 'Total time for this run: ', t2-t1
    if rank == 0: print '------------------------------'
