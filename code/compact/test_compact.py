import numpy as np
from mpi4py import MPI

from mpi_util import *
from compact import CompactFiniteDifferenceSolver

if __name__ == "__main__":
    comm = MPI.COMM_WORLD 
    da = DA(comm, (32, 8, 16), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*np.sin(y) + z*np.cos(y*x)
    dfdx_true = np.sin(y) - z*y*np.sin(y*x)
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd = CompactFiniteDifferenceSolver(da)
    dfdx = cfd.dfdx(f, dx)
    
    import matplotlib.pyplot as plt
    if da.rank == 0:
        plt.plot(dfdx_true[3, 3, :])
        plt.plot(dfdx[3, 3, :], '-o')
        plt.savefig('temp.png')
