from createDA import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [1, 1, 3]
local_dims = [1, 3, 3]
nz, ny, nx = local_dims

da = create_da(proc_sizes, local_dims)

a = np.empty(local_dims, dtype=np.float64)
a.fill(rank)
b = np.empty([nz+2, ny+2, nx+2], dtype=np.float64)

da.global_to_local(a, b)

print rank

def test_center():
    # test center:
    if rank == 1:
        assert(np.all(da.left_recv_halo == 0))
        assert(np.all(da.left_send_halo == 1))
        assert(np.all(da.right_send_halo == 1))
        assert(np.all(da.right_recv_halo == 2))

def test_left_and_right():
    # test left and right:
    if rank == 0:
        assert(np.all(da.right_recv_halo == 1))

    if rank == 2:
        assert(np.all(da.left_recv_halo == 1))

MPI.Finalize()
