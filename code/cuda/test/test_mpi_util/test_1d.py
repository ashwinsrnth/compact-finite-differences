from createDA import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [1, 1, 3]
local_dims = [1, 3, 3]
nz, ny, nx = local_dims

da = create_da(proc_sizes, local_dims)
a_d = da.create_global_vector()
a_d.fill(rank)
b_d = da.create_local_vector()

da.global_to_local(a_d, b_d)

def test_center():
    # test center:
    if rank == 1:
        assert(np.all(da.left_recv_halo.get() == 0))
        assert(np.all(da.left_send_halo.get() == 1))
        assert(np.all(da.right_send_halo.get() == 1))
        assert(np.all(da.right_recv_halo.get() == 2))

def test_left_and_right():
    # test left and right:
    if rank == 0:
        assert(np.all(da.right_recv_halo.get() == 1))

    if rank == 2:
        assert(np.all(da.left_recv_halo.get() == 1))

MPI.Finalize()
