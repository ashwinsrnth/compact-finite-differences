import sys
sys.path.append('..')
import numpy as np
from mpi4py import MPI
from gpuDA import *
from compact import CompactFiniteDifferenceSolver
from numpy.testing import *
from pycuda import autoinit
import pycuda.gpuarray as gpuarray    

comm = MPI.COMM_WORLD 
da_regular = DA(comm, (32, 32, 32), (2, 2, 2), 1)
da_irregular = DA(comm, (64, 32, 32), (2, 2, 2), 1)
cfd_regular = CompactFiniteDifferenceSolver(da_regular)
cfd_irregular = CompactFiniteDifferenceSolver(da_irregular)

def test_dfdx_sine_regular():
    x, y, z = DA_arange(da_regular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    f_d = gpuarray.to_gpu(f)
    x_d = da_regular.create_global_vector()
    f_local_d = da_regular.create_local_vector()
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd_regular.dfdx(f_d, dx, x_d, f_local_d)
    dfdx = x_d.get()
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdx_sine_irregular():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = np.sin(x) 
    f_d = gpuarray.to_gpu(f)
    x_d = da_irregular.create_global_vector()
    f_local_d = da_irregular.create_local_vector()
    dfdx_true = np.cos(x) 
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd_irregular.dfdx(f_d, dx, x_d, f_local_d)
    dfdx = x_d.get()
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'

def test_dfdx_xyz():
    x, y, z = DA_arange(da_irregular, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
    f = x*y*z
    f_d = gpuarray.to_gpu(f)
    x_d = da_irregular.create_global_vector()
    f_local_d = da_irregular.create_local_vector()
    dfdx_true = y*z
    dx = x[0, 0, 1] - x[0, 0, 0]
    cfd_irregular.dfdx(f_d, dx, x_d, f_local_d)
    dfdx = x_d.get()
    assert_almost_equal(dfdx_true, dfdx, decimal=2)
    if comm.Get_rank() == 0:
        print 'pass'
'''
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
'''
if __name__ == "__main__":
    test_dfdx_sine_regular()
    test_dfdx_sine_irregular()
    test_dfdx_xyz()
    #test_dfdy_sine_regular()
    #test_dfdy_xyz()
    #test_dfdz_xyz()
    MPI.Finalize()
