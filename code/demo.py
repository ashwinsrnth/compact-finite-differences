import numpy as np
from mpi4py import MPI
from mpi_util import *
from compact import CompactFiniteDifferenceSolver
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD 
da = DA(comm, (8, 32, 16), (2, 2, 2), 1)
x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
f = y*np.cos(x*y) + z*y
dfdx_true = (-y**2)*np.sin(x*y)
dfdy_true = -(x*y)*np.sin(x*y) + np.cos(x*y) + z

dy = y[0, 1, 0] - y[0, 0, 0]
dx = x[0, 0, 1] - x[0, 0, 0]
cfd = CompactFiniteDifferenceSolver(da)

dfdx = cfd.dfdx(f, dx)
dfdx_global = np.zeros([16, 64, 32], dtype=np.float64)
dfdx_true_global = np.zeros([16, 64, 32], dtype=np.float64)

DA_gather_blocks(da, dfdx, dfdx_global)
DA_gather_blocks(da, dfdx_true, dfdx_true_global)

if comm.Get_rank() == 0:
    plt.plot(dfdx_global[4, :, 8], 'o-')
    plt.plot(dfdx_true_global[4, :, 8])

dfdy = cfd.dfdy(f, dy)
dfdy_global = np.zeros([16, 64, 32], dtype=np.float64)
dfdy_true_global = np.zeros([16, 64, 32], dtype=np.float64)

DA_gather_blocks(da, dfdy, dfdy_global)
DA_gather_blocks(da, dfdy_true, dfdy_true_global)

if comm.Get_rank() == 0:
    plt.plot(dfdy_global[4, :, 8], 'o-')
    plt.plot(dfdy_true_global[4, :, 8])
    plt.savefig('demo.png')
