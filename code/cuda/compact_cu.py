from mpi4py import MPI
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy.linalg import solve_banded
import kernels
from near_toeplitz_cu import *
from pthomas_cu import *
from mpi_util import *

class CompactFiniteDifferenceSolver:

    def __init__(self, da, use_gpu=False):
        '''
        :param da: DA object carrying the grid information
        :type da: mpi_util.DA
        :param use_gpu: set True if using GPU
        :type use_gpu: bool
        '''
        self.da = da
        self.use_gpu = use_gpu
        self.init_cu()
        self.init_solvers()

    def dfdx(self, f, dx):
        '''
        :param f: The 3-d array with function values
        :type f: numpy.ndarray
        :param dx: Spacing in x-direction
        :type dx: float
        '''
        r_d = self.compute_RHS(self.x_line_da, f, dx)
        x_UH, x_LH = self.solve_secondary_systems(self.x_line_da)
        self.x_primary_solver.solve(r_d, [1, 1])
        alpha, beta = self.solve_reduced_system(self.x_line_da, x_UH, x_LH, r_d, self.x_reduced_solver)
        self.sum_solutions(self.x_line_da, r_d, x_UH, x_LH, alpha, beta)
        dfdx = r_d.get()
        return dfdx 
    
    def dfdy(self, f, dy):
        f_T = f.transpose(0, 2, 1).copy()
        r_d = self.compute_RHS(self.y_line_da, f_T, dy)
        x_UH, x_LH = self.solve_secondary_systems(self.y_line_da)
        self.y_primary_solver.solve(r_d, [1, 1])
        alpha, beta = self.solve_reduced_system(self.y_line_da, x_UH, x_LH, r_d, self.y_reduced_solver)
        self.sum_solutions(self.y_line_da, r_d, x_UH, x_LH, alpha, beta)
        dfdy = r_d.get()
        dfdy = dfdy.transpose(0, 2, 1).copy()
        return dfdy 

    def dfdz(self, f, dz):
        f_T = f.transpose(1, 2, 0).copy()
        r_d = self.compute_RHS(self.z_line_da, f_T, dz)
        x_UH, x_LH = self.solve_secondary_systems(self.z_line_da)
        self.z_primary_solver.solve(r_d, [1, 1])
        alpha, beta = self.solve_reduced_system(self.z_line_da, x_UH, x_LH, r_d, self.z_reduced_solver)
        self.sum_solutions(self.z_line_da, r_d, x_UH, x_LH, alpha, beta)
        dfdz = r_d.get()
        dfdz = dfdz.transpose(2, 0, 1).copy()
        return dfdz

    def compute_RHS(self, line_da, f, dx):
        f_local = line_da.create_local_vector()
        line_da.global_to_local(f, f_local)
        f_d = gpuarray.to_gpu(f_local)
        x_d = gpuarray.zeros((line_da.nz, line_da.ny, line_da.nx),
                dtype=np.float64)
        self.compute_RHS_kernel.prepare([np.intp, np.intp, np.float64, np.intc, np.intc])
        self.compute_RHS_kernel.prepared_call((line_da.nx/8, line_da.ny/8, line_da.nz/8), (8, 8, 8),
                    f_d.gpudata, x_d.gpudata, np.float64(dx),
                        np.int32(line_da.rank), np.int32(line_da.size))
        return x_d
    
    def sum_solutions(self, line_da, x_R_d, x_UH, x_LH, alpha, beta):
        x_UH_d = gpuarray.to_gpu(x_UH)
        x_LH_d = gpuarray.to_gpu(x_LH)
        alpha_d = gpuarray.to_gpu(alpha)
        beta_d = gpuarray.to_gpu(beta)
        self.sum_solutions_kernel.prepare([np.intp, np.intp,
                np.intp, np.intp, np.intp,
                    np.intc, np.intc, np.intc])
        self.sum_solutions_kernel.prepared_call(
                (line_da.nx/8, line_da.ny/8, line_da.nz/8),
                    (8, 8, 8),
                        x_R_d.gpudata, x_UH_d.gpudata,
                        x_LH_d.gpudata, alpha_d.gpudata, beta_d.gpudata,
                            np.int32(line_da.nx),
                            np.int32(line_da.ny),
                            np.int32(line_da.nz))

    def solve_reduced_system(self, line_da, x_UH, x_LH, x_R_d, reduced_solver):
        nz, ny, nx = line_da.nz, line_da.ny, line_da.nx
        line_rank = line_da.rank
        line_size = line_da.size
        
        x_UH_line = np.zeros(2*line_size, dtype=np.float64)
        x_LH_line = np.zeros(2*line_size, dtype=np.float64)
        line_da.gather(
                [np.array([x_UH[0], x_UH[-1]]), 2, MPI.DOUBLE],
                [x_UH_line, 2, MPI.DOUBLE])
        line_da.gather(
                [np.array([x_LH[0], x_LH[-1]]), 2, MPI.DOUBLE],
                [x_LH_line, 2, MPI.DOUBLE])

        lengths = np.ones(line_size)
        displacements = np.arange(0, 2*line_size, 2)
        start_z, start_y, start_x = 0, 0, displacements[line_rank]
        subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, 2*line_size],
                            [nz, ny, 2], [start_z, start_y, start_x])
        subarray = subarray_aux.Create_resized(0, 8)
        subarray.Commit()
        
        x_R_faces_d = gpuarray.zeros((nz, ny, 2), np.float64)
        self.copy_faces_kernel.prepare([np.intp, np.intp,
            np.intc, np.intc, np.intc])
        self.copy_faces_kernel.prepared_call((1, ny/16, nz/16), (1, 16, 16),
                x_R_d.gpudata, x_R_faces_d.gpudata,
                    np.int32(nx), np.int32(ny), np.int32(nz))
        x_R_faces = x_R_faces_d.get()
        x_R_faces_line = np.zeros([nz, ny, 2*line_size], dtype=np.float64)
        line_da.gatherv([x_R_faces, MPI.DOUBLE],
                [x_R_faces_line, lengths, displacements, subarray])
        
        if line_rank == 0:
            a_reduced = np.zeros(2*line_size, dtype=np.float64)
            b_reduced = np.zeros(2*line_size, dtype=np.float64)
            c_reduced = np.zeros(2*line_size, dtype=np.float64)
            a_reduced[0::2] = -1.
            a_reduced[1::2] = x_UH_line[1::2]
            b_reduced[0::2] = x_UH_line[0::2]
            b_reduced[1::2] = x_LH_line[1::2]
            c_reduced[0::2] = x_LH_line[0::2]
            c_reduced[1::2] = -1.
            a_reduced[0], c_reduced[0] = 0.0, 0.0
            b_reduced[0] = 1.0
            a_reduced[-1], c_reduced[-1] = 0.0, 0.0
            b_reduced[-1] = 1.0
            a_reduced[1] = 0.
            c_reduced[-2] = 0.
            x_R_faces_line[:, :, 0] = 0.0
            x_R_faces_line[:, :, -1] = 0.0
            
            a_reduced_d = gpuarray.to_gpu(a_reduced)
            b_reduced_d = gpuarray.to_gpu(b_reduced)
            c_reduced_d = gpuarray.to_gpu(c_reduced)
            c2_reduced_d = gpuarray.to_gpu(c_reduced)
            d_reduced_d = gpuarray.to_gpu(-x_R_faces_line)
            reduced_solver.solve(a_reduced_d, b_reduced_d,
                    c_reduced_d, c2_reduced_d, d_reduced_d)
            params = d_reduced_d.get()
        else:
            params = None
        
        params_local = np.zeros([nz, ny, 2], dtype=np.float64)
        line_da.scatterv([params, lengths, displacements, subarray],
                [params_local, MPI.DOUBLE])
        alpha = params_local[:, :, 0].copy()
        beta = params_local[:, :, 1].copy()
        return alpha, beta
        
    def solve_secondary_systems(self, line_da):
        nz, ny, nx = line_da.nz, line_da.ny, line_da.nx
        line_rank = line_da.rank
        line_size = line_da.size

        a = np.ones(nx, dtype=np.float64)*(1./4)
        b = np.ones(nx, dtype=np.float64)
        c = np.ones(nx, dtype=np.float64)*(1./4)
        r_UH = np.zeros(nx, dtype=np.float64)
        r_LH = np.zeros(nx, dtype=np.float64)

        if line_rank == 0:
            c[0] =  2.0
            a[0] = 0.0

        if line_rank == line_size-1:
            a[-1] = 2.0
            c[-1] = 0.0

        r_UH[0] = -a[0]
        r_LH[-1] = -c[-1]

        x_UH = scipy_solve_banded(a, b, c, r_UH)
        x_LH = scipy_solve_banded(a, b, c, r_LH)
        return x_UH, x_LH

    def setup_reduced_solver(self, line_da):
       return PThomas((line_da.nz, line_da.ny, 2*line_da.npx))

    def setup_primary_solver(self, line_da):
        line_rank = line_da.rank
        line_size = line_da.size
        coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
        if line_rank == 0:
            coeffs[1] = 2.
        if line_rank == line_size-1:
            coeffs[-2] = 2.
        return NearToeplitzSolver((line_da.nz, line_da.ny, line_da.nx), coeffs)

    def init_cu(self):
        self.compute_RHS_kernel, self.sum_solutions_kernel, self.copy_faces_kernel, = kernels.get_funcs(
                'kernels.cu', 'computeRHS', 'sumSolutions', 'copyFaces')
                 
    def init_solvers(self):
        self.x_line_da = self.da.get_line_DA(0)
        self.y_line_da = self.da.get_line_DA(1)
        self.z_line_da = self.da.get_line_DA(2)
        self.x_primary_solver = self.setup_primary_solver(self.x_line_da)
        self.y_primary_solver = self.setup_primary_solver(self.y_line_da)
        self.z_primary_solver = self.setup_primary_solver(self.z_line_da)
        self.x_reduced_solver = self.setup_reduced_solver(self.x_line_da)
        self.y_reduced_solver = self.setup_reduced_solver(self.y_line_da)
        self.z_reduced_solver = self.setup_reduced_solver(self.z_line_da)

def scipy_solve_banded(a, b, c, rhs):
    '''
    Solve the tridiagonal system described
    by a, b, c, and rhs.
    a: lower off-diagonal array (first element ignored)
    b: diagonal array
    c: upper off-diagonal array (last element ignored)
    rhs: right hand side of the system
    '''
    l_and_u = (1, 1)
    ab = np.vstack([np.append(0, c[:-1]),
                    b,
                    np.append(a[1:], 0)])
    x = solve_banded(l_and_u, ab, rhs)
    return x
