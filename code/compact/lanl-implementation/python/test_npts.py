import numpy as np
from npts import *
from numpy.testing import assert_allclose
from mpi4py import MPI
import tools

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

#np.random.seed(1241851)

npx = 2
npy = 2
npz = 2

comm = comm.Create_cart([npz, npy, npx], reorder=False)

NX = 72
NY = 36
NZ = 36

nx = NX/npx
ny = NY/npy
nz = NZ/npz


beta, gamma = precompute_beta_gam_dfdx(comm, NX, NY, NZ)
r = np.random.rand(nz, ny, nx)
x = dfdx_parallel(comm, beta, gamma, r)

if rank == 0:
    x_full = np.zeros([NZ, NY, NX], dtype=np.float64)
else:
    x_full = None
if rank == 0:
    r_full = np.random.rand(NZ, NY, NX)
else:
    r_full = None

tools.gather_3D(comm, r, r_full)
tools.gather_3D(comm, x, x_full)

if rank == 0:
    for i in range(NZ):
        for j in range(NY):
            print i, j
            a = np.ones(NX, dtype=np.float64)*(1./4)
            b = np.ones(NX, dtype=np.float64)
            c = np.ones(NX, dtype=np.float64)*(1./4)
            a[-1] = 2.0
            c[0] = 2.0
            x = scipy_solve_banded(a, b, c, r_full[i, j, :])
            assert_allclose(x, x_full[i, j, :])
            print x, x_full[i, j, :]
