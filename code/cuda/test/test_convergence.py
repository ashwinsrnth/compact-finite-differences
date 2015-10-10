# Order of convergence test:
# The interior operator itself yields 4th order
# accuracy, and the boundary operator
# is third order accurate.
# The fourth order convergence
# of the numerical method is still preserved,
# unless the boundaries are a primary source of
# error

import sys
sys.path.append('..')
import numpy as np
from mpi4py import MPI
from gpuDA import *
from compact import CompactFiniteDifferenceSolver
from numpy.testing import *
from pycuda import autoinit
import pycuda.gpuarray as gpuarray    

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()

mean_errs = []
max_errs = []
sizes = [16, 32, 64, 128, 256]

for i, N in enumerate(sizes):
    da = DA(comm, (N, N, N), (2, 2, 2), 1)
    line_da = da.get_line_DA(0)
    cfd = CompactFiniteDifferenceSolver(line_da)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) + np.cos(y*x) + z*x
    f_d = gpuarray.to_gpu(f)
    x_d = da.create_global_vector()
    f_local_d = da.create_local_vector()
    dfdx_true = np.cos(x) + -y*np.sin(y*x) + z
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd.dfdx(f_d, dx, x_d, f_local_d)
    dfdx = x_d.get()
    err = np.abs(dfdx-dfdx_true)/np.max(abs(dfdx))
    mean_err = np.mean(np.abs(dfdx-dfdx_true)/np.max(abs(dfdx)))
    max_err = np.max(np.abs(dfdx-dfdx_true)/np.max(abs(dfdx)))
    mean_errs.append(mean_err)
    max_errs.append(max_err)

if rank == 0:
    print 
    for i, N in enumerate(sizes[1:]):
        print "Mean err(N={0})/ Mean err(N={1}) = {2}".format(sizes[i], sizes[i+1], mean_errs[i]/mean_errs[i+1])
    print 
    for i, N in enumerate(sizes[1:]):
        print "Max err(N={0})/ Max err(N={1}) = {2}".format(sizes[i], sizes[i+1], max_errs[i]/max_errs[i+1])
MPI.Finalize()
