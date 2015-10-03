import numpy as np
from mpi4py import MPI
from mpi_util import *
from compact import CompactFiniteDifferenceSolver
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD 
da = DA(comm, (8, 32, 16), (2, 2, 2), 1)
x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
f = y*np.cos(x*y) + np.sin(z)*y
dfdx_true = (-y**2)*np.sin(x*y)
dfdy_true = -(x*y)*np.sin(x*y) + np.cos(x*y) + np.sin(z)
dfdz_true = y*np.cos(z)

dz = z[1, 0, 0] - z[0, 0, 0]
dy = y[0, 1, 0] - y[0, 0, 0]
dx = x[0, 0, 1] - x[0, 0, 0]

cfd = CompactFiniteDifferenceSolver(da)

dfdx = cfd.dfdx(f, dx)
dfdx_global = np.zeros([16, 64, 32], dtype=np.float64)
dfdx_true_global = np.zeros([16, 64, 32], dtype=np.float64)

DA_gather_blocks(da, dfdx, dfdx_global)
DA_gather_blocks(da, dfdx_true, dfdx_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, 32), dfdx_global[8, 32, :], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, 32), dfdx_true_global[8, 32, :])

dfdy = cfd.dfdy(f, dy)
dfdy_global = np.zeros([16, 64, 32], dtype=np.float64)
dfdy_true_global = np.zeros([16, 64, 32], dtype=np.float64)

DA_gather_blocks(da, dfdy, dfdy_global)
DA_gather_blocks(da, dfdy_true, dfdy_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, 64), dfdy_global[8, :, 16], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, 64), dfdy_true_global[8, :, 16])

dfdz = cfd.dfdz(f, dz)
dfdz_global = np.zeros([16, 64, 32], dtype=np.float64)
dfdz_true_global = np.zeros([16, 64, 32], dtype=np.float64)

DA_gather_blocks(da, dfdz, dfdz_global)
DA_gather_blocks(da, dfdz_true, dfdz_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, 16), dfdz_global[:, 32, 8], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, 16), dfdz_true_global[:, 32, 8])
    plt.savefig('demo.svg')
