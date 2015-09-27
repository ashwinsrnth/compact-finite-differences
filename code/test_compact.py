import numpy as np
from mpi4py import MPI
from mpi_util import *
from compact import CompactFiniteDifferenceSolver

from numpy.testing import *

def test_dfdx_sine_regular():
    comm = MPI.COMM_WORLD 
    da = DA(comm, (8, 8, 8), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd = CompactFiniteDifferenceSolver(da)
    dfdx = cfd.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)

def test_dfdx_sine_irregular():
    comm = MPI.COMM_WORLD 
    da = DA(comm, (8, 32, 16), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd = CompactFiniteDifferenceSolver(da)
    dfdx = cfd.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)

def test_dfdx_xyz():
    comm = MPI.COMM_WORLD 
    da = DA(comm, (8, 32, 16), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*y*z
    dfdx_true = y*z
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd = CompactFiniteDifferenceSolver(da)
    dfdx = cfd.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)

def test_dfdy_sine_regular():
    comm = MPI.COMM_WORLD 
    da = DA(comm, (8, 8, 8), (2, 2, 2), 1)
    x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(y) 
    dfdy_true = np.cos(y) 
    dy = y[0, 1, 0] - y[0, 0, 0]
    cfd = CompactFiniteDifferenceSolver(da)
    dfdy = cfd.dfdy(f, dy)
    assert_almost_equal(dfdy_true, dfdy, decimal=2)

if __name__ == "__main__":
    test_dfdx_sine_regular()
    test_dfdx_sine_irregular()
    test_dfdx_xyz()
    test_dfdy_sine_regular()
