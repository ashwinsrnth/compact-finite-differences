from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded
import mpiDA
import tridiagonal

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

def dfdx(comm, f, dx):

    comm.Barrier()
    t_start = MPI.Wtime()

    rank = comm.Get_rank()
    size = comm.Get_size()
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]
    nz, ny, nx = f.shape
    NZ, NY, NX = nz*npz, ny*npy, nx*npx

    da = mpiDA.DA(comm, [nz, ny, nx], [npz, npy, npx], 1)
    batch_solver = tridiagonal.BatchTridiagonalSolver(comm)

    f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
    d = np.zeros([nz, ny, nx], dtype=np.float64)

    da.global_to_local(f, f_local)

    comm.Barrier()
    t1 = MPI.Wtime()

    d[:, :, :] = (3./4)*(f_local[1:-1, 1:-1, 2:] - f_local[1:-1, 1:-1, :-2])/dx
    if mx == 0:
        d[:, :, 0] = (1./(2*dx))*(-5*f[:, :, 0] + 4*f[:, :, 1] + f[:, :, 2])
    if mx == npx-1:
        d[:, :, -1] = -(1./(2*dx))*(-5*f[:, :, -1] + 4*f[:, :, -2] + f[:, :, -3])

    comm.Barrier()
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

    comm.Barrier()
    t1 = MPI.Wtime()

    x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
    x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Time to solve the UH and LH local systems: ', t2-t1

    t1 = MPI.Wtime()

    x_R = batch_solver.solve(a_line_local, b_line_local, c_line_local, d, nz*ny, nx)
    x_R = x_R.reshape([nz, ny, nx])

    comm.Barrier()
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

    comm.Barrier()
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

    comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
        [x_LH_global, lengths, displacements, MPI.DOUBLE], root=line_root)

    comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
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

    comm.Barrier()

    # since we're using a subarray, set lengths to 1:
    lengths[line_processes] = 1

    comm.Gatherv([x_R_faces, MPI.DOUBLE],
        [x_R_global, lengths, displacements, subarray], root=line_root)

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Gathering the data to the line_root: ', t2-t1

    #---------------------------------------------------------------------------
    # assemble and solve the reduced systems at all ranks mx=0
    # to compute the transfer parameters

    comm.Barrier()
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

        params = batch_solver.solve(a_reduced, b_reduced, c_reduced, -d_reduced, nz*ny, 2*npx)
        params = params.reshape([nz, ny, 2*npx])
    else:
        params = None

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Assembling and solving the reduced system: ', t2-t1

    #------------------------------------------------------------------------------
    # scatter the parameters back

    params_local = np.zeros([nz, ny, 2], dtype=np.float64)

    comm.Scatterv([params, lengths, displacements, subarray],
        [params_local, MPI.DOUBLE], root=line_root)

    alpha = params_local[:, :, 0]
    beta = params_local[:, :, 1]

    comm.Barrier()
    t1 = MPI.Wtime()

    # note the broadcasting below!
    dfdx_local = x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)

    comm.Barrier()
    t2 = MPI.Wtime()

    if rank == 0: print 'Computing the sum of solutions: ', t2-t1

    comm.Barrier()
    t_end = MPI.Wtime()

    if rank == 0: print 'Total time: ', t_end-t_start

    return dfdx_local

def dfdy(comm, f, dy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]
    nz, ny, nx = f.shape
    NZ, NY, NX = nz*npz, ny*npy, nx*npx

    da = mpiDA.DA(comm, [nz, ny, nx], [npz, npy, npx], 1)

    f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
    d = np.zeros([nz, ny, nx], dtype=np.float64)

    da.global_to_local(f, f_local)

    d[:, :, :] = (3./4)*(f_local[1:-1, 2:, 1:-1] - f_local[1:-1, :-2, 1:-1])/dy
    if my == 0:
        d[:, 0, :] = (1./(2*dy))*(-5*f[:, 0, :] + 4*f[:, 1, :] + f[:, 2, :])
    if my == npy-1:
        d[:, -1, :] = -(1./(2*dy))*(-5*f[:, -1, :] + 4*f[:, -2, :] + f[:, -3, :])

    #---------------------------------------------------------------------------
    # create the LHS for the tridiagonal system of the compact difference scheme:
    a_line_local = np.ones(ny, dtype=np.float64)*(1./4)
    b_line_local = np.ones(ny, dtype=np.float64)
    c_line_local = np.ones(ny, dtype=np.float64)*(1./4)

    if my == 0:
        c_line_local[0] = 2.0
        a_line_local[0] = 0.0

    if my == npy-1:
        a_line_local[-1] = 2.0
        c_line_local[-1] = 0.0

    #------------------------------------------------------------------------------
    # each processor computes x_R, x_LH_line and x_UH_line:
    r_LH_line = np.zeros(ny, dtype=np.float64)
    r_UH_line = np.zeros(ny, dtype=np.float64)
    r_LH_line[-1] = -c_line_local[-1]
    r_UH_line[0] = -a_line_local[0]

    x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
    x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

    x_R = np.zeros([nz, nx, ny], dtype=np.float64)
    for i in range(nz):
        for j in range(nx):
            x_R[i,j,:] = scipy_solve_banded(a_line_local, b_line_local, c_line_local, d[i,:,j])
    comm.Barrier()

    #---------------------------------------------------------------------------
    # the first and last elements in x_LH and x_UH,
    # and also the first and last "faces" in x_R,
    # need to be gathered at rank 0:

    if rank == 0:
        x_LH_global = np.zeros([2*size], dtype=np.float64)
        x_UH_global = np.zeros([2*size], dtype=np.float64)
    else:
        x_LH_global = None
        x_UH_global = None

    lengths = np.ones(size)*2
    displacements = np.arange(0, 2*size, 2).reshape([npz, npy, npx]).transpose(0, 2, 1).copy().ravel()

    comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
        [x_LH_global, lengths, displacements, MPI.DOUBLE])

    comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
        [x_UH_global, lengths, displacements, MPI.DOUBLE])

    if rank == 0:
        x_R_global = np.zeros([nz, nx, 2*size], dtype=np.float64)
    else:
        x_R_global = None

    start_z, start_y, start_x = 0, 0, displacements[rank]
    subarray_aux = MPI.DOUBLE.Create_subarray([nz, nx, 2*size],
                        [nz, nx, 2], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    x_R_faces = np.zeros([nz, nx, 2], dtype=np.float64)
    x_R_faces[:, :, 0] = x_R[:, :, 0].copy()
    x_R_faces[:, :, 1] = x_R[:, :, -1].copy()

    comm.Barrier()
    comm.Gatherv([x_R_faces, MPI.DOUBLE],
        [x_R_global, np.ones(size, dtype=np.int), displacements, subarray], root=0)

    #---------------------------------------------------------------------------
    # assemble and solve the reduced matrix to compute the transfer parameters

    if rank == 0:
        a_reduced = np.zeros([2*size], dtype=np.float64)
        b_reduced = np.zeros([2*size], dtype=np.float64)
        c_reduced = np.zeros([2*size], dtype=np.float64)
        d_reduced = np.zeros([nz, nx, 2*size], dtype=np.float64)
        d_reduced[...] = x_R_global

        a_reduced[0::2] = -1.
        a_reduced[1::2] = x_UH_global[1::2]
        b_reduced[0::2] = x_UH_global[0::2]
        b_reduced[1::2] = x_LH_global[1::2]
        c_reduced[0::2] = x_LH_global[0::2]
        c_reduced[1::2] = -1.

        a_reduced[0::2*npy], c_reduced[0::2*npy], d_reduced[:, :, 0::2*npy] = 0.0, 0.0, 0.0
        b_reduced[0::2*npy] = 1.0
        a_reduced[-1::-2*npy], c_reduced[-1::-2*npy], d_reduced[:, :, -1::-2*npy] = 0.0, 0.0, 0.0
        b_reduced[-1::-2*npy] = 1.0

        a_reduced[1::2*npy] = 0.
        c_reduced[-2::-2*npy] = 0.

        params = np.zeros([nz, nx, 2*size])
        for i in range(nz):
            for j in range(nx):
                params[i, j, :] = scipy_solve_banded(a_reduced, b_reduced, c_reduced, -d_reduced[i, j, :])


        np.testing.assert_allclose(params[:, :, 0::2*npy], 0)
        np.testing.assert_allclose(params[:, :, -1::-2*npy], 0)

    else:
        params = None

    comm.Barrier()

    #------------------------------------------------------------------------------
    # scatter the parameters back

    params_local = np.zeros([nz, nx, 2], dtype=np.float64)

    comm.Scatterv([params, np.ones(size, dtype=int), displacements, subarray],
        [params_local, MPI.DOUBLE])

    alpha = params_local[:, :, 0]
    beta = params_local[:, :, 1]

    # note the broadcasting below!
    comm.Barrier()
    dfdy_local = x_R.transpose(0, 2, 1) + np.einsum('ij,k->ikj', alpha, x_UH_line) + np.einsum('ij,k->ikj', beta, x_LH_line)
    comm.Barrier()

    return dfdy_local

def dfdz(comm, f, dz):
    rank = comm.Get_rank()
    size = comm.Get_size()
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]
    nz, ny, nx = f.shape
    NZ, NY, NX = nz*npz, ny*npy, nx*npx

    da = mpiDA.DA(comm, [nz, ny, nx], [npz, npy, npx], 1)

    f_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
    d = np.zeros([nz, ny, nx], dtype=np.float64)

    da.global_to_local(f, f_local)

    d[:, :, :] = (3./4)*(f_local[2:, 1:-1, 1:-1] - f_local[:-2, 1:-1, 1:-1])/dz
    if mz == 0:
        d[0, : :] = (1./(2*dz))*(-5*f[0, :, :] + 4*f[1, :, :] + f[2, :, :])
    if mz == npz-1:
        d[-1, :, :] = -(1./(2*dz))*(-5*f[-1, :, :] + 4*f[-2, :, :] + f[-3, :, :])

    #---------------------------------------------------------------------------
    # create the LHS for the tridiagonal system of the compact difference scheme:
    a_line_local = np.ones(nz, dtype=np.float64)*(1./4)
    b_line_local = np.ones(nz, dtype=np.float64)
    c_line_local = np.ones(nz, dtype=np.float64)*(1./4)

    if mz == 0:
        c_line_local[0] = 2.0
        a_line_local[0] = 0.0

    if mz == npz-1:
        a_line_local[-1] = 2.0
        c_line_local[-1] = 0.0

    #------------------------------------------------------------------------------
    # each processor computes x_R, x_LH_line and x_UH_line:
    r_LH_line = np.zeros(nz, dtype=np.float64)
    r_UH_line = np.zeros(nz, dtype=np.float64)
    r_LH_line[-1] = -c_line_local[-1]
    r_UH_line[0] = -a_line_local[0]

    x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
    x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

    x_R = np.zeros([ny, nx, nz], dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            x_R[i,j,:] = scipy_solve_banded(a_line_local, b_line_local, c_line_local, d[:, i, j])
    comm.Barrier()

    #---------------------------------------------------------------------------
    # the first and last elements in x_LH and x_UH,
    # and also the first and last "faces" in x_R,
    # need to be gathered at rank 0:

    if rank == 0:
        x_LH_global = np.zeros([2*size], dtype=np.float64)
        x_UH_global = np.zeros([2*size], dtype=np.float64)
    else:
        x_LH_global = None
        x_UH_global = None

    lengths = np.ones(size)*2
    displacements = np.arange(0, 2*size, 2).reshape([npz, npy, npx]).transpose(2, 0, 1).copy().ravel()

    comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
        [x_LH_global, lengths, displacements, MPI.DOUBLE])

    comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
        [x_UH_global, lengths, displacements, MPI.DOUBLE])

    if rank == 0:
        x_R_global = np.zeros([ny, nx, 2*size], dtype=np.float64)
    else:
            x_R_global = None

    start_z, start_y, start_x = 0, 0, displacements[rank]
    subarray_aux = MPI.DOUBLE.Create_subarray([ny, nx, 2*size],
                        [ny, nx, 2], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    x_R_faces = np.zeros([ny, nx, 2], dtype=np.float64)
    x_R_faces[:, :, 0] = x_R[:, :, 0].copy()
    x_R_faces[:, :, 1] = x_R[:, :, -1].copy()

    comm.Barrier()
    comm.Gatherv([x_R_faces, MPI.DOUBLE],
        [x_R_global, np.ones(size, dtype=np.int), displacements, subarray], root=0)

    #---------------------------------------------------------------------------
    # assemble and solve the reduced matrix to compute the transfer parameters

    if rank == 0:
        a_reduced = np.zeros([2*size], dtype=np.float64)
        b_reduced = np.zeros([2*size], dtype=np.float64)
        c_reduced = np.zeros([2*size], dtype=np.float64)
        d_reduced = np.zeros([ny, nx, 2*size], dtype=np.float64)
        d_reduced[...] = x_R_global

        a_reduced[0::2] = -1.
        a_reduced[1::2] = x_UH_global[1::2]
        b_reduced[0::2] = x_UH_global[0::2]
        b_reduced[1::2] = x_LH_global[1::2]
        c_reduced[0::2] = x_LH_global[0::2]
        c_reduced[1::2] = -1.

        a_reduced[0::2*npz], c_reduced[0::2*npz], d_reduced[:, :, 0::2*npz] = 0.0, 0.0, 0.0
        b_reduced[0::2*npz] = 1.0
        a_reduced[-1::-2*npz], c_reduced[-1::-2*npz], d_reduced[: , :, -1::-2*npz] = 0.0, 0.0, 0.0
        b_reduced[-1::-2*npz] = 1.0

        a_reduced[1::2*npz] = 0.
        c_reduced[-2::-2*npz] = 0.

        params = np.zeros([ny, nx, 2*size])
        for i in range(ny):
            for j in range(nx):
                params[i, j, :] = scipy_solve_banded(a_reduced, b_reduced, c_reduced, -d_reduced[i, j, :])

        np.testing.assert_allclose(params[:, :, 0::2*npz], 0)
        np.testing.assert_allclose(params[:, :, -1::-2*npz], 0)

    else:
        params = None

    comm.Barrier()

    #------------------------------------------------------------------------------
    # scatter the parameters back

    params_local = np.zeros([ny, nx, 2], dtype=np.float64)

    comm.Scatterv([params, np.ones(size, dtype=int), displacements, subarray],
        [params_local, MPI.DOUBLE])

    alpha = params_local[:, :, 0]
    beta = params_local[:, :, 1]

    comm.Barrier()
    dfdz_local = x_R.transpose(2, 0, 1).copy() + np.einsum('ij,k->kij', alpha, x_UH_line) + np.einsum('ij,k->kij', beta, x_LH_line)
    comm.Barrier()

    return dfdz_local
