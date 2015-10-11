from createDA import *
from numpy.testing import *
import pycuda.gpuarray as gpuarray

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
        a_d = self.da.create_global_vector()
        a_d.fill(self.rank)

        # fill b with ones
        b_d = self.da.create_local_vector()
        b_d.fill(1.0)
        
        self.da.global_to_local(a_d, b_d)

        a = a_d.get()
        b = b_d.get()

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
        b_d = gpuarray.to_gpu(b)

        # a is empty
        a_d = self.da.create_global_vector()
        
        self.da.local_to_global(b_d, a_d)
        a = a_d.get()
        b = b_d.get()

        # test ltog:
        if self.rank == 0:
            assert_equal(a, b[2:-2,2:-2,2:-2])
    
    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
