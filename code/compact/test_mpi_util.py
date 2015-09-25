from mpi_util import *
import numpy as np
from mpi4py import MPI
from numpy.testing import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert(size == 8)

def test_DA_arange():
    da = DA(comm, [5, 5, 5], [2, 2, 2], 1)
    x, y, z = DA_arange(da, (1., 10.), (1., 10.), (1., 10.))
    if rank == 1:
        for i in range(5):
            for j in range(5):
                assert_equal(x[i, j, :], np.arange(6, 11))

if __name__ == "__main__":
    test_DA_arange()
