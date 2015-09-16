from mpi4py import MPI
import numpy as np

def scatter_3D(comm, x_global, x_local):
    assert (isinstance(comm, MPI.Cartcomm))

    mz, my, mx = comm.Get_topo()[2]
    npz, npy, npx = comm.Get_topo()[0]
    nz, ny, nx = x_local.shape
    NZ, NY, NX = npz*nz, npy*ny, npx*nx
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    start_z, start_y, start_x = mz*nz, my*ny, mx*nx
    subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                        [nz, ny, nx], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
    sendbuf = [start_index, MPI.INT]
    displs = np.zeros(size, dtype=np.int)
    recvbuf = [displs, MPI.INT]
    comm.Gather(sendbuf, recvbuf, root=0)
    comm.Barrier()

    comm.Scatterv([x_global, np.ones(size, dtype=np.int), displs, subarray],
        [x_local, MPI.DOUBLE], root=0)

    subarray.Free()

def gather_3D(comm, x_local, x_global):
    assert (isinstance(comm, MPI.Cartcomm))

    mz, my, mx = comm.Get_topo()[2]
    npz, npy, npx = comm.Get_topo()[0]
    nz, ny, nx = x_local.shape
    NZ, NY, NX = npz*nz, npy*ny, npx*nx
    size = comm.Get_size()
    rank = comm.Get_rank()

    start_z, start_y, start_x = mz*nz, my*ny, mx*nx
    subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                        [nz, ny, nx], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
    sendbuf = [start_index, MPI.INT]
    displs = np.zeros(size, dtype=np.int)
    recvbuf = [displs, MPI.INT]
    comm.Gather(sendbuf, recvbuf, root=0)
    comm.Barrier()

    comm.Gatherv([x_local, MPI.DOUBLE],
        [x_global, np.ones(size, dtype=np.int), displs, subarray], root=0)

    subarray.Free()
