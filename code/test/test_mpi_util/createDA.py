import sys
sys.path.append('../..')
from mpi_util import *

def create_da(proc_sizes, local_dims):
    '''
    Convenience function: creates a DA for running
    the tests
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    npz, npy, npx = proc_sizes    
    nz, ny, nx = local_dims
    assert(npx*npy*npz == size)
    comm = comm.Create_cart([npz, npy, npx], reorder=True)
    da = DA(comm, [nz, ny, nx], [npz, npy, npx], 1)
     
    return da
