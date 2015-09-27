from mpi4py import MPI
import pyopencl.array as cl_array
import pyopencl as cl
import numpy as np
from scipy.linalg import solve_banded

import kernels
from near_toeplitz import *
from pthomas import *
from mpi_util import *


class CompactFiniteDifferenceSolver:

    def __init__(self, da, use_gpu = False):
        '''
        :param da: DA object carrying the grid information
        :type da: mpi_util.DA
        :param use_gpu: set True if using GPU
        :type use_gpu: bool
        '''
        self.da = da
        self.use_gpu = use_gpu
        self.init_cl()
    
    def dfdx(self, f, dx):
        '''
        :param f: The 3-d array with function values
        :type f: numpy.ndarray
        :param dx: Spacing in x-direction
        :type dx: float
        '''
        line_da = self.da.get_line_DA(0)
        f_local = line_da.create_local_vector()
        
        self.setup_primary_solver(line_da)
        line_da.global_to_local(f, f_local)
        rhs = self.compute_RHS_dfdx(line_da, f_local, dx)
        x_UH, x_LH = self.solve_secondary_systems(line_da)
        x_R = self.solve_primary_systems(rhs)
        alpha, beta = self.solve_reduced_system(line_da, x_UH, x_LH, x_R)
        dfdx = self.sum_solutions(x_R, x_UH, x_LH, alpha, beta)
        return dfdx 
    
    def dfdy(self, f, dy):
        line_da = self.da.get_line_DA(1)
        f_T = f.transpose(0, 2, 1).copy()
        f_local = line_da.create_local_vector()
        
        self.setup_primary_solver(line_da)
        line_da.global_to_local(f_T, f_local)
        rhs = self.compute_RHS_dfdx(line_da, f_local, dy)
        x_UH, x_LH = self.solve_secondary_systems(line_da)
        x_R = self.solve_primary_systems(rhs)
        alpha, beta = self.solve_reduced_system(line_da, x_UH, x_LH, x_R)
        dfdy = self.sum_solutions(x_R, x_UH, x_LH, alpha, beta)
        dfdy = dfdy.transpose(0, 2, 1).copy()
        return dfdy 

    def compute_RHS_dfdx(self, line_da, f_local, dx):
        f_d = cl_array.to_device(self.queue, f_local)
        x_d = cl_array.Array(self.queue, (line_da.nz, line_da.ny, line_da.nx),
                dtype=np.float64)
        self.compute_RHS_kernel(self.queue, (line_da.nx, line_da.ny, line_da.nz),
                None, f_d.data, x_d.data, np.float64(dx),
                    np.int32(line_da.rank), np.int32(line_da.size))
        rhs = x_d.get()
        return rhs 
    
    def sum_solutions(self, x_R, x_UH, x_LH, alpha, beta):
        return (x_R + 
                np.einsum('ij,k->ijk', alpha, x_UH) +
                np.einsum('ij,k->ijk', beta, x_LH))
        
    def solve_primary_systems(self, rhs):
        r_d = cl_array.to_device(self.queue, rhs)
        self.primary_solver.solve(r_d.data, [1, 1])
        x_R = r_d.get()
        return x_R

    def solve_reduced_system(self, line_da, x_UH, x_LH, x_R):
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
        
        x_R_faces = x_R[:, :, [0, -1]].copy()
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
            reduced_solver = PThomas(self.ctx, self.queue, [nz, ny, 2*line_size],
                    a_reduced, b_reduced, c_reduced)
            x_R_faces_line[:, :, 0] = 0.0
            x_R_faces_line[:, :, -1] = 0.0
            d_reduced_d = cl_array.to_device(self.queue, -x_R_faces_line)
            reduced_solver.solve(d_reduced_d.data)
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

    def setup_primary_solver(self, line_da):
        line_rank = line_da.rank
        line_size = line_da.size
        coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
        if line_rank == 0:
            coeffs[1] = 2.
        if line_rank == line_size-1:
            coeffs[-2] = 2.
        self.primary_solver = NearToeplitzSolver(self.ctx, self.queue,
                (line_da.nz, line_da.ny, line_da.nx), coeffs)

    def init_cl(self):
        self.platform = cl.get_platforms()[0]
        if self.use_gpu:
            ngpus = len(self.platform.get_devices())
            self.device = self.platform.get_devices()[self.da.rank%ngpus]
        else:
            self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        self.compute_RHS_kernel, = kernels.get_funcs(self.ctx, 'kernels.cl',
                'computeRHSdfdx')
                 

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
