import numpy as np
from mpi4py import MPI
from mpi_util import *
from compact_cu import CompactFiniteDifferenceSolver
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD 

nz = 32
ny = 32
nx = 32

da = DA(comm, (nz, ny, nx), (1, 1, 1), 1)
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
dfdx_global = np.zeros([nz, ny, nx], dtype=np.float64)
dfdx_true_global = np.zeros([nz, ny, nx], dtype=np.float64)

DA_gather_blocks(da, dfdx, dfdx_global)
DA_gather_blocks(da, dfdx_true, dfdx_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, nx), dfdx_global[nz/2, ny/2, :], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, nx), dfdx_true_global[nz/2, ny/2, :])

dfdy = cfd.dfdy(f, dy)
dfdy_global = np.zeros([nz, ny, nx], dtype=np.float64)
dfdy_true_global = np.zeros([nz, ny, nx], dtype=np.float64)

DA_gather_blocks(da, dfdy, dfdy_global)
DA_gather_blocks(da, dfdy_true, dfdy_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, ny), dfdy_global[nz/2, :, nx/2], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, ny), dfdy_true_global[nz/2, :, nx/2])

dfdz = cfd.dfdz(f, dz)
dfdz_global = np.zeros([nz, ny, nx], dtype=np.float64)
dfdz_true_global = np.zeros([nz, ny, nx], dtype=np.float64)

DA_gather_blocks(da, dfdz, dfdz_global)
DA_gather_blocks(da, dfdz_true, dfdz_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, nz), dfdz_global[:, ny/2, nx/2], 'o-')
    plt.plot(np.linspace(0, 2*np.pi, nz), dfdz_true_global[:, ny/2, nx/2])
    plt.savefig('demo.svg')
