from createDA import *
from numpy.testing import *

class TestDOF:
    @classmethod
    def setup_class(cls):
        comm = MPI.COMM_WORLD
        cls.rank = comm.Get_rank()
        cls.size = comm.Get_size()
        proc_sizes = [1, 1, 3]
        local_dims = [1, 3, 3]
        nz, ny, nx = local_dims
        cls.da = create_da(proc_sizes, local_dims, sw=2)

    def test_center(self):
        a = self.da.create_global_vector()
        a.fill(self.rank)
        b = self.da.create_local_vector()
        b.fill(1.0)
        self.da.global_to_local(a, b)

        # test center:
        if self.rank == 1:
            assert_equal(self.da.left_recv_halo, 0)
            assert_equal(self.da.left_send_halo, 1)
            assert_equal(self.da.right_send_halo, 1)
            assert_equal(self.da.right_recv_halo, 2)
    
    def test_sides(self):
        a = self.da.create_global_vector()
        a.fill(self.rank)
        b = self.da.create_local_vector()
        self.da.global_to_local(a, b)

        # test left and right:
        if self.rank == 0:
            assert_equal(self.da.right_recv_halo, 1)

        if self.rank == 2:
            assert_equal(self.da.left_recv_halo, 1)
    
    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
