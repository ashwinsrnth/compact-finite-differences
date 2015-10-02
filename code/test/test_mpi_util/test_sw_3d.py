from createDA import *
from numpy.testing import *

class TestGpuDA3d:
    @classmethod
    def setup_class(cls): 
        cls.comm = MPI.COMM_WORLD
        cls.rank = cls.comm.Get_rank()
        cls.size = cls.comm.Get_size()
        assert(cls.size == 27)
        cls.proc_sizes = [3, 3, 3]
        cls.local_dims = [4, 4, 4]
        cls.da = create_da(cls.proc_sizes, cls.local_dims, sw=2)
    
    def test_gtol(self):

        nz, ny, nx = self.local_dims
        
        # fill a with rank
        a = self.da.create_global_vector()
        a.fill(self.rank)

        # fill b with ones
        b = self.da.create_local_vector()
        b.fill(1.0)
        
        self.da.global_to_local(a, b)

        # test gtol at the center
        if self.rank == 13:
            assert_equal(b[2:-2,2:-2,:2], 12)
            assert_equal(b[2:-2,2:-2,-2:], 14)
            assert_equal(b[2:-2,:2,2:-2], 10)
            assert_equal(b[2:-2,-2:,2:-2], 16)
            assert_equal(b[:2,2:-2,2:-2], 4)
            assert_equal(b[-2:,2:-2,2:-2], 22)
        
        # test that the boundaries remain unaffected:
        if self.rank == 22:
            # since we initially filled b with ones
            assert_equal(b[-2:,:,:], 1)
    
    def test_ltog(self):

        nz, ny, nx = self.local_dims

        # fill b with a sequence
        b = np.ones([nz+4, ny+4, nx+4], dtype=np.float64)
        b = b*np.arange((nx+4)*(ny+4)*(nz+4)).reshape([nz+4, ny+4, nx+4])

        # a is empty
        a = self.da.create_global_vector()
        print b.shape, a.shape
        self.da.local_to_global(b, a)

        # test ltog:
        if self.rank == 0:
            assert_equal(a, b[2:-2,2:-2,2:-2])
    
    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
