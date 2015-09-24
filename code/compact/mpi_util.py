import numpy as np
from mpi4py import MPI

def MPI_get_line(comm, direction):
    '''
    Get intracommunicator composed
    of only processes in the "line"
    along the given direction.

    direction:

    0: x
    1: y
    2: z
    '''
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]
    ranks_matrix = np.arange(npz*npy*npx).reshape([npz, npy, npx])
    global_group = comm.Get_group()
    if direction == 0:
        line_group = global_group.Incl(ranks_matrix[mz, my, :])
    elif direction == 1:
        line_group = global_group.Incl(ranks_matrix[mz, :, mx])
    else:
        line_group = global_group.Incl(ranks_matrix[:, my, mx])
    line_comm = comm.Create(line_group)
    return line_comm

def face_type(line_comm, shape):
    '''
    '''
    nz, ny, nx = shape
    npx = line_comm.Get_size()
    displacements = np.arange(0, 2*npx, 2)
    line_rank = line_comm.Get_rank()
    start_z, start_y, start_x = 0, 0, displacements[line_rank]
    subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, 2*npx],
                        [nz, ny, 2], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()
    return subarray


