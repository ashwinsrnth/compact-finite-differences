from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded
import mpiDA
import kernels
import pyopencl as cl

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


class CompactFiniteDifferenceSolver:

    def __init__(self, ctx, queue, comm, sizes):
        '''
        sizes: tuple/list
            The global size of the problem [NZ, NY, NX]
        '''
        self.ctx = ctx
        self.queue = queue
        self.comm = comm
        self.prg = kernels.get_kernels(self.ctx)

        self.NZ, self.NY, self.NX = sizes
        self.mz, self.my, self.mx = self.comm.Get_topo()[2] # this proc's ID in each direction
        self.npz, self.npy, self.npx = self.comm.Get_topo()[0] # num procs in each direction
        self.nz, self.ny, self.nx = self.NZ/self.npz, self.NY/self.npy, self.NX/self.npx # local sizes

        self.prg = kernels.get_kernels(self.ctx)
        self.da = mpiDA.DA(self.comm.Clone(), [self.nz, self.ny, self.nx], [self.npz, self.npy, self.npx], 1)

        self.f_local = np.zeros([self.nz+2, self.ny+2, self.nx+2], dtype=np.float64)

    def dfdx(self, f, dx):
        '''
        Get the local x-derivative given
        the local portion of f.
        '''
        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        mz, my, mx = self.mz, self.my, self.mx
        NZ, NY, NX = self.NZ, self.NY, self.NX
        nz, ny, nx = self.nz, self.ny, self.nx
        npz, npy, npx = self.npz, self.npy, self.npx
        assert(f.shape == (nz, ny, nx))

        t_start = MPI.Wtime()

        #---------------------------------------------------------------------------
        # compute the RHS of the system

        self.da.global_to_local(f, self.f_local)

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t1 = MPI.Wtime()

        d = self.computeRHS(self.f_local, dx, mx, npx)

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Time to create RHS: ', t2-t1

        #---------------------------------------------------------------------------
        # create the LHS for the tridiagonal system of the compact difference scheme:
        a_line_local = np.ones(nx, dtype=np.float64)*(1./4)
        b_line_local = np.ones(nx, dtype=np.float64)
        c_line_local = np.ones(nx, dtype=np.float64)*(1./4)

        if mx == 0:
            c_line_local[0] = 2.0
            a_line_local[0] = 0.0

        if mx == npx-1:
            a_line_local[-1] = 2.0
            c_line_local[-1] = 0.0

        #------------------------------------------------------------------------------
        # each processor computes x_R, x_LH_line and x_UH_line:
        r_LH_line = np.zeros(nx, dtype=np.float64)
        r_UH_line = np.zeros(nx, dtype=np.float64)
        r_LH_line[-1] = -c_line_local[-1]
        r_UH_line[0] = -a_line_local[0]

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t1 = MPI.Wtime()

        x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
        x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Time to solve the UH and LH local systems: ', t2-t1

        t1 = MPI.Wtime()

        x_R = self.batch_solver(a_line_local, b_line_local, c_line_local, d, nz*ny, nx)
        x_R = x_R.reshape([nz, ny, nx])

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Time to solve the RHS local system: ', t2-t1

        #---------------------------------------------------------------------------
        # the first and last elements in x_LH and x_UH,
        # and also the first and last "faces" in x_R,
        # need to be gathered at the rank
        # that is at the beginning of the line (line_root)

        # to avoid a separate communicator for this purpose,
        # we use Gatherv with lengths and displacements as 0
        # for all processes not in the line

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t1 = MPI.Wtime()

        if mx == 0:
            x_LH_global = np.zeros([2*npx], dtype=np.float64)
            x_UH_global = np.zeros([2*npx], dtype=np.float64)
        else:
            x_LH_global = None
            x_UH_global = None

        procs_matrix = np.arange(size, dtype=int).reshape([npz, npy, npx])
        line_root = procs_matrix[mz, my, 0]         # the root procs of this line
        line_processes = procs_matrix[mz, my, :]    # all procs in this line

        # initialize lengths and displacements to 0
        lengths = np.zeros(size)
        displacements = np.zeros(size)

        # only the processes in the line get lengths and displacements
        lengths[line_processes] = 2
        displacements[line_processes] = range(0, 2*npx, 2)

        self.comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
            [x_LH_global, lengths, displacements, MPI.DOUBLE], root=line_root)

        self.comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
            [x_UH_global, lengths, displacements, MPI.DOUBLE], root=line_root)

        if mx == 0:
            x_R_global = np.zeros([nz, ny, 2*npx], dtype=np.float64)
        else:
            x_R_global = None

        start_z, start_y, start_x = 0, 0, displacements[rank]
        subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, 2*npx],
                            [nz, ny, 2], [start_z, start_y, start_x])
        subarray = subarray_aux.Create_resized(0, 8)
        subarray.Commit()

        x_R_faces = np.zeros([nz, ny, 2], dtype=np.float64)
        x_R_faces[:, :, 0] = x_R[:, :, 0].copy()
        x_R_faces[:, :, 1] = x_R[:, :, -1].copy()

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()

        # since we're using a subarray, set lengths to 1:
        lengths[line_processes] = 1

        self.comm.Gatherv([x_R_faces, MPI.DOUBLE],
            [x_R_global, lengths, displacements, subarray], root=line_root)

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Gathering the data to the line_root: ', t2-t1

        #---------------------------------------------------------------------------
        # assemble and solve the reduced systems at all ranks mx=0
        # to compute the transfer parameters

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t1 = MPI.Wtime()

        if mx == 0:
            a_reduced = np.zeros([2*npx], dtype=np.float64)
            b_reduced = np.zeros([2*npx], dtype=np.float64)
            c_reduced = np.zeros([2*npx], dtype=np.float64)
            d_reduced = np.zeros([nz, ny, 2*npx], dtype=np.float64)
            d_reduced[...] = x_R_global

            a_reduced[0::2] = -1.
            a_reduced[1::2] = x_UH_global[1::2]
            b_reduced[0::2] = x_UH_global[0::2]
            b_reduced[1::2] = x_LH_global[1::2]
            c_reduced[0::2] = x_LH_global[0::2]
            c_reduced[1::2] = -1.

            a_reduced[0], c_reduced[0], d_reduced[:, :, 0] = 0.0, 0.0, 0.0
            b_reduced[0] = 1.0
            a_reduced[-1], c_reduced[-1], d_reduced[:, :, -1] = 0.0, 0.0, 0.0
            b_reduced[-1] = 1.0

            a_reduced[1] = 0.
            c_reduced[-2] = 0.

            params = self.batch_solver(a_reduced, b_reduced, c_reduced, -d_reduced, nz*ny, 2*npx)
            params = params.reshape([nz, ny, 2*npx])
        else:
            params = None

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Assembling and solving the reduced system: ', t2-t1

        #------------------------------------------------------------------------------
        # scatter the parameters back

        params_local = np.zeros([nz, ny, 2], dtype=np.float64)

        self.comm.Scatterv([params, lengths, displacements, subarray],
            [params_local, MPI.DOUBLE], root=line_root)

        alpha = params_local[:, :, 0]
        beta = params_local[:, :, 1]

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t1 = MPI.Wtime()

        # note the broadcasting below!
        dfdx_local = x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t2 = MPI.Wtime()

        if rank == 0: print 'Computing the sum of solutions: ', t2-t1

        cl.enqueue_barrier(self.queue)
        self.comm.Barrier()
        t_end = MPI.Wtime()

        if rank == 0: print 'Total time: ', t_end-t_start

        return dfdx_local


    def computeRHS(self, f_local, dx, mx, npx):
        '''
        Compute the right hand side for
        the x-derivative
        '''

        nz, ny, nx = f_local[1:-1, 1:-1, 1:-1].shape
        d = np.zeros([nz, ny, nx], dtype=np.float64)

        f_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, (nz+2)*(ny+2)*(nx+2)*8)
        d_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nz*ny*nx*8)

        cl.enqueue_copy(self.queue, f_g, f_local)

        evt = self.prg.computeRHSdfdx(self.queue, [nx, ny, nz], None,
            f_g, d_g, np.float64(dx), np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(mx), np.int32(npx))

        cl.enqueue_copy(self.queue, d, d_g)

        return d

    def batch_solver(self, a, b, c, d, num_systems, system_size):
        dfdx = np.zeros(num_systems*system_size, dtype=np.float64)
        a_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        b_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        c_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        c2_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, system_size*8)
        d_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)
        dfdx_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)

        evt1 = cl.enqueue_copy(self.queue, a_g, a)
        evt2 = cl.enqueue_copy(self.queue, b_g, b)
        evt3 = cl.enqueue_copy(self.queue, c_g, c)
        evt4 = cl.enqueue_copy(self.queue, d_g, d)
        evt5 = cl.enqueue_copy(self.queue, c2_g, c)

        evt = self.prg.compactTDMA(self.queue, [num_systems], None,
            a_g, b_g, c_g, d_g, dfdx_g, c2_g,
                np.int32(system_size))
        evt.wait()

        evt = cl.enqueue_copy(self.queue, dfdx, dfdx_g)
        evt.wait()

        return dfdx
