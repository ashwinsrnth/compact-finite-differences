import tools
from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded

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
    rank = comm.Get_rank()
    size = comm.Get_size()
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]
    if rank == 0:
        NZ, NY, NX = f.shape
    else:
        NZ, NY, NX = None, None, None
    NZ = comm.bcast(NZ)
    NY = comm.bcast(NY)
    NX = comm.bcast(NX)
    nz, ny, nx = NZ/npz, NY/npy, NX/npx

    if rank == 0:
        d = np.zeros([NZ, NY, NX], dtype=np.float64)
        d[:, :, 1:-1] = (3./4)*(f[:, :, 2:] - f[:, :, :-2])/dx
        d[:, :, 0] = (1./(2*dx))*(-5*f[:,:, 0] + 4*f[:, :, 1] + f[:, :, 2])
        d[:, :, -1] = -(1./(2*dx))*(-5*f[:, :, -1] + 4*f[:, :, -2] + f[:, :, -3])
    else:
        d = None

    f_local = np.zeros([nz, ny, nx], dtype=np.float64)
    d_local = np.zeros([nz, ny, nx], dtype=np.float64)
    tools.scatter_3D(comm, f, f_local)
    tools.scatter_3D(comm, d, d_local)

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

    x_LH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_LH_line)
    x_UH_line = scipy_solve_banded(a_line_local, b_line_local, c_line_local, r_UH_line)

    x_R = np.zeros([nz, ny, nx], dtype=np.float64)
    for i in range(nz):
        for j in range(ny):
            x_R[i,j,:] = scipy_solve_banded(a_line_local, b_line_local, c_line_local, d_local[i,j,:])
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
    displacements = np.arange(0, 2*size, 2)

    comm.Gatherv([np.array([x_LH_line[0], x_LH_line[-1]]), MPI.DOUBLE],
        [x_LH_global, lengths, displacements, MPI.DOUBLE])

    comm.Gatherv([np.array([x_UH_line[0], x_UH_line[-1]]), MPI.DOUBLE],
        [x_UH_global, lengths, displacements, MPI.DOUBLE])

    if rank == 0:
        x_R_global = np.zeros([nz, ny, 2*size], dtype=np.float64)
    else:
        x_R_global = None

    start_z, start_y, start_x = 0, 0, 2*rank
    subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, 2*size],
                        [nz, ny, 2], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    x_R_faces = np.zeros([nz, ny, 2], dtype=np.float64)
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
        d_reduced = np.zeros([nz, ny, 2*size], dtype=np.float64)
        d_reduced[...] = x_R_global

        a_reduced[0::2] = -1.
        a_reduced[1::2] = x_UH_global[1::2]
        b_reduced[0::2] = x_UH_global[0::2]
        b_reduced[1::2] = x_LH_global[1::2]
        c_reduced[0::2] = x_LH_global[0::2]
        c_reduced[1::2] = -1.

        a_reduced[0::2*npx], c_reduced[0::2*npx], d_reduced[:,:,0::2*npx] = 0.0, 0.0, 0.0
        b_reduced[0::2*npx] = 1.0
        a_reduced[-1::-2*npx], c_reduced[-1::-2*npx], d_reduced[:,:,-1::-2*npx] = 0.0, 0.0, 0.0
        b_reduced[-1::-2*npx] = 1.0

        a_reduced[1::2*npx] = 0.
        c_reduced[-2::-2*npx] = 0.

        params = np.zeros([nz, ny, 2*size])
        for i in range(nz):
            for j in range(ny):
                params[i, j, :] = scipy_solve_banded(a_reduced, b_reduced, c_reduced, -d_reduced[i, j, :])

        np.testing.assert_allclose(params[:, :, 0::2*npx], 0)
        np.testing.assert_allclose(params[:, :, -1::-2*npx], 0)

    else:
        params = None

    comm.Barrier()

    #------------------------------------------------------------------------------
    # scatter the parameters back

    params_local = np.zeros([nz, ny, 2], dtype=np.float64)

    comm.Scatterv([params, np.ones(size, dtype=int), displacements, subarray],
        [params_local, MPI.DOUBLE])

    alpha = params_local[:,:,0]
    beta = params_local[:,:,1]

    # note the broadcasting below!
    comm.Barrier()
    dfdx_local = x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
    comm.Barrier()

    if rank == 0:
        dfdx = np.zeros([NZ, NY, NX], dtype=np.float64)
    else:
        dfdx = None

    tools.gather_3D(comm, dfdx_local, dfdx)
    return dfdx
