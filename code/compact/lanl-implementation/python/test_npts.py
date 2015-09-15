import numpy as np
from npts import *
from numpy.testing import assert_allclose
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

assert(nprocs == 4)
system_size = 32
local_size = system_size/nprocs

if rank == 0:
    r = np.random.rand(system_size)
else:
    r = None

if rank == 0:
    beta_global = np.zeros(system_size, dtype=np.float64)
    gam_global = np.zeros(system_size, dtype=np.float64)
    x_global = np.zeros(system_size, dtype=np.float64)
else:
    beta_global = None
    gam_global = None
    x_global = None

r_local = np.zeros(local_size, dtype=np.float64)
comm.Scatter([r, MPI.DOUBLE], [r_local, MPI.DOUBLE])

x_local = solve_parallel(comm, r_local)
comm.Gather([x_local, MPI.DOUBLE], [x_global, MPI.DOUBLE])

if rank == 0:
    a = np.ones(system_size, dtype=np.float64)*(1./3)
    b = np.ones(system_size, dtype=np.float64)
    c = np.ones(system_size, dtype=np.float64)*(1./3)
    x = scipy_solve_banded(a, b, c, r)

    print 'Testing with 4 processors, system size = {0}'.format(system_size)
    assert_allclose(x, x_global)
    print '.OK'
