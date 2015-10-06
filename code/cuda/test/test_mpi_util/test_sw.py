from createDA import *
from numpy.testing import *

class TestDOF:
    @classmethod
    def setup_class(cls):
        comm = MPI.COMM_WORLD
        cls.rank = comm.Get_rank()
        cls.size = comm.Get_size()
        proc_sizes = [1, 1, 3]
        local_dims = [3, 3, 3]
        nz, ny, nx = local_dims
        cls.da = create_da(proc_sizes, local_dims, sw=2)

    def test_center(self):
        a_d = self.da.create_global_vector()
        a_d.fill(self.rank)
        b_d = self.da.create_local_vector()
        b_d.fill(1.0)
        self.da.global_to_local(a_d, b_d)

        # test center:
        if self.rank == 1:
            assert_equal(self.da.left_recv_halo.get(), 0)
            assert_equal(self.da.left_send_halo.get(), 1)
            assert_equal(self.da.right_send_halo.get(), 1)
            assert_equal(self.da.right_recv_halo.get(), 2)
    
    def test_sides(self):
        a_d = self.da.create_global_vector()
        a_d.fill(self.rank)
        b_d = self.da.create_local_vector()
        b_d.fill(1.0)
        self.da.global_to_local(a_d, b_d)

        # test left and right:
        if self.rank == 0:
            assert_equal(self.da.right_recv_halo.get(), 1)

        if self.rank == 2:
            assert_equal(self.da.left_recv_halo.get(), 1)
    
    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
