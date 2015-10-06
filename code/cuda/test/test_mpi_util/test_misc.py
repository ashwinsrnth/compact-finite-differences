import sys
sys.path.append('../../')
from gpuDA import *
import numpy as np
from mpi4py import MPI
from numpy.testing import *

from pycuda import autoinit
import pycuda.gpuarray as gpuarray
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert(size == 8)

def test_DA_arange():
    da = DA(comm, (5, 5, 5), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (1., 10.), (1., 10.), (1., 10.))
    if rank == 1:
        for i in range(5):
            for j in range(5):
                assert_equal(x[i, j, :], np.arange(6, 11))
    print 'pass'

def test_DA_get_line_DA():
    da = DA(comm, [5, 5, 5], [2, 2, 2], 1)
    line_da = da.get_line_DA(0)
    assert(line_da.npx == 2)
    assert(line_da.npy == 1)
    assert(line_da.npz == 1)
    print 'pass'

def test_DA_gather():
    da = DA(comm, [5, 5, 5], [2, 2, 2], 1)
    a_d = da.create_global_vector()
    a_d.fill(rank)
    a_root_d = gpuarray.zeros(8, dtype=np.float64)
    da.gatherv([a_d.gpudata.as_buffer(a_d.nbytes), 1, MPI.DOUBLE],
                [a_root_d.gpudata.as_buffer(a_root_d.nbytes), np.ones(8), range(8), MPI.DOUBLE])
    if rank == 0:
        assert_equal(a_root_d.get(), np.arange(8))
    else:
        assert_equal(a_root_d.get(), 0)
    print 'pass'

if __name__ == "__main__":
    test_DA_arange()
    test_DA_get_line_DA()
    test_DA_gather()
    MPI.Finalize()
