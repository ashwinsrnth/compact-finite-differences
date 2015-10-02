import sys
sys.path.append('..')
import numpy as np
from mpi4py import MPI
from mpi_util import *
from compact import CompactFiniteDifferenceSolver
from numpy.testing import *

comm = MPI.COMM_WORLD 
da_regular = DA(comm, (8, 8, 8), (2, 2, 2), 1)
da_irregular = DA(comm, (8, 32, 16), (2, 2, 2), 1)
cfd_regular = CompactFiniteDifferenceSolver(da_regular)
cfd_irregular = CompactFiniteDifferenceSolver(da_irregular)

def test_dfdx_sine_regular():
    x, y, z = DA_arange(da_regular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    dfdx = cfd_regular.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdx_sine_irregular():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    dfdx = cfd_irregular.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdx_xyz():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*y*z
    dfdx_true = y*z
    dx = x[0, 0, 1] - x[0, 0, 0]
    dfdx = cfd_irregular.dfdx(f, dx)
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdy_sine_regular():
    x, y, z = DA_arange(da_regular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(y) 
    dfdy_true = np.cos(y) 
    dy = y[0, 1, 0] - y[0, 0, 0]
    dfdy = cfd_regular.dfdy(f, dy)
    assert_almost_equal(dfdy_true, dfdy, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdy_xyz():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*y*z 
    dfdy_true = x*z 
    dy = y[0, 1, 0] - y[0, 0, 0]
    dfdy = cfd_irregular.dfdy(f, dy)
    assert_almost_equal(dfdy_true, dfdy, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'
 
def test_dfdz_xyz():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*y*z**2
    dfdz_true = 2*z*x*y 
    dz = z[1, 0, 0] - z[0, 0, 0]
    dfdz = cfd_irregular.dfdz(f, dz)
    assert_almost_equal(dfdz_true, dfdz, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

if __name__ == "__main__":
    test_dfdx_sine_regular()
    test_dfdx_sine_irregular()
    test_dfdx_xyz()
    test_dfdy_sine_regular()
    test_dfdy_xyz()
    test_dfdz_xyz()
