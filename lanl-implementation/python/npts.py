from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded

def solve_parallel(comm, rhs_local):
    '''
    Solve the tridiagonal system:
      1       2       .       .       .
      1/4     1.      1/4     .       .
      .       1/4     1       1/4     .
      .       .       .       .       .
      .       .       .       .       .
      .       .       .       .       .

    in parallel,
    for some right-hand side `rhs_local`.
    Returns `x_local`, the local part of the solution.
    '''

    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    local_size = rhs_local.size
    system_size = local_size*nprocs
    beta_local, gam_local = precompute_beta_gam(comm, system_size)
    x_local = nonperiodic_tridiagonal_solver(comm, beta_local, gam_local, rhs_local, system_size)
    return x_local

# an interface to SciPy's tridiagonal
# solver - used for testing
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


# The following functions work for lines along the
# x-direction only. Consider a "dir" parameter
# that will generalize them.

def get_line_info(comm):
    '''
    Get the root and number of processes
    in the x-direction.
    '''
    size = comm.Get_size()
    rank = comm.Get_rank()
    mz, my, mx = comm.Get_topo()[2]
    npz, npy, npx = comm.Get_topo()[0]
    procs_matrix = np.arange(size, dtype=int).reshape([npz, npy, npx])
    line_root = procs_matrix[mz, my, 0]         # the root procs of this line
    line_processes = list(procs_matrix[mz, my, :])    # all procs in this line
    return line_root, line_processes

def line_subarray(comm, shape, subarray_length):
    '''
    For each block in a line in the x-direction, shaped [nz, ny, nx],
    get a "subarray" MPI datatype shaped [nz, ny, subarray_length].
    Also get the "lengths" and "displacements" parameters for
    scatter and gather operations with these subarrays
    '''
    nz, ny, nx = shape
    npz, npy, npx = comm.Get_topo()[0]
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # initialize lengths and displacements to 0
    lengths = np.zeros(nprocs, dtype=int)
    displacements = np.zeros(nprocs, dtype=int)

    line_root, line_processes = get_line_info(comm)
    line_nprocs = len(line_processes)
    line_last = line_processes[-1]

    # only the processes in the line get lengths and displacements
    lengths[line_processes] = subarray_length
    displacements[line_processes] = range(0, subarray_length*npx, subarray_length)

    start_z, start_y, start_x = 0, 0, int(displacements[rank])
    subarray_aux = MPI.DOUBLE.Create_subarray([nz, ny, npx],
                        [nz, ny, subarray_length], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()
    return lengths, displacements, subarray

def line_bcast(comm, buf, root):
    '''
    Perform a broadcast of "buf" from root "root" to all processes
    in the line
    '''
    rank = comm.Get_rank()
    line_root, line_processes = get_line_info(comm)
    line_nprocs = len(line_processes)
    line_last = line_processes[-1]
    messages = []

    line_processes.remove(root)

    if rank == root:
        for dest in line_processes:
            comm.Isend([buf, buf.size, MPI.DOUBLE], dest=dest, tag=dest*10)
    if rank != root:
        message = (comm.Irecv([buf, buf.size, MPI.DOUBLE], source=root, tag=rank*10))
        MPI.Request.Wait(message)

def line_allgather_faces(comm, x, x_faces, face):

    '''
    gather
    '''

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nz, ny, nx = x.shape
    npz, npy, npx = comm.Get_topo()[0]

    line_root, line_processes = get_line_info(comm)

    lengths, displacements, subarray = line_subarray(comm, (nz, ny, nx), 1)
    comm.Barrier()

    if face == 'last':
        x_face = x[:, :, -1].copy()
    else:
        x_face = x[:, :, 0].copy()

    comm.Gatherv([x_face, MPI.DOUBLE], [x_faces, lengths, displacements, subarray], root=line_root)

    line_bcast(comm, x_faces, line_root)

def line_allgather(comm, x, x_line):

    '''
    Allgather scalar elements along a line in the x-direction
    '''

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    npz, npy, npx = comm.Get_topo()[0]

    # initialize lengths and displacements to 0
    lengths = np.zeros(nprocs, dtype=int)
    displacements = np.zeros(nprocs, dtype=int)

    line_root, line_processes = get_line_info(comm)
    line_nprocs = len(line_processes)
    line_last = line_processes[-1]

    # only the processes in the line get lengths and displacements
    lengths[line_processes] = 1
    displacements[line_processes] = range(npx)

    comm.Barrier()

    comm.Gatherv([x, 1, MPI.DOUBLE], [x_line, lengths, displacements, MPI.DOUBLE], root=line_root)

    line_bcast(comm, x_line, line_root)


def precompute_beta_gam_dfdx(comm, NX, NY, NZ):
    # Pre-computes the beta and gam required
    # by the tridiagonal solver
    # The tridiagonal system has:
    # This needs to be done in a pipelined
    # manner, but only once.
    # a[i] = 1./4
    # b[i] = 1.0
    # c[i] = 1./4
    '''
    comm: Cartcomm
    direction: 0=z, 1=y, 2=x
    system_size: size in "direction" direction.
    '''
    rank = comm.Get_rank()
    mz, my, mx = comm.Get_topo()[2]
    npz, npy, npx = comm.Get_topo()[0]
    nz, ny, nx = NZ/npz, NY/npy, NX/npx

    line_root, line_processes = get_line_info(comm)
    line_last = line_root + npx - 1
    line_nprocs = len(line_processes)

    beta_local = np.zeros(nx, dtype=np.float64)
    gamma_local = np.zeros(nx, dtype=np.float64)
    last_beta = np.zeros(1, dtype=np.float64)

    # do a serial hand-off:
    for r in range(line_root, line_last+1):
        if rank == r:
            if rank == line_root:
                beta_local[0] = 1.0
                gamma_local[0] = 0.0
            else:
                comm.Recv([last_beta, 1, MPI.DOUBLE], source=rank-1, tag=10)
                beta_local[0] = 1./(1. - (1./4)*last_beta*(1./4))
                gamma_local[0] = last_beta*(1./4)

            for i in range(1, nx):
                if rank == line_root and i == 1:
                    gamma_local[i] = beta_local[i-1]*2
                else:
                    gamma_local[i] = beta_local[i-1]*(1./4)

                if rank == line_last and i == nx-1:
                    beta_local[i] = 1./(1. - (2.0)*beta_local[i-1]*(1./4))
                elif rank == line_root and i == 1:
                    beta_local[i] = 1./(1. - (2.0)*beta_local[i-1]*(1./4))
                else:
                    beta_local[i] = 1./(1. - (1./4)*beta_local[i-1]*(1./4))
            if rank != line_last:
                comm.Send([beta_local[-1:], 1, MPI.DOUBLE], dest=rank+1, tag=10)

    return beta_local, gamma_local


def dfdx_parallel(comm, beta_local, gam_local, r):

    nz, ny, nx = r.shape

    x = np.zeros_like(r, dtype=np.float64)
    u = np.zeros_like(r, dtype=np.float64)

    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mz, my, mx = comm.Get_topo()[2]
    npz, npy, npx = comm.Get_topo()[0]
    NZ, NY, NX = nz*npz, ny*npy, nx*npx

    line_root, line_processes = get_line_info(comm)
    line_last = line_root + npx - 1
    line_nprocs = len(line_processes)

    #############
    # L-R sweep
    #############

    assert(npz*npy*npx == nprocs)

    phi = np.zeros([nz, ny, nx], dtype=np.float64)
    psi = np.zeros([nz, ny, nx], dtype=np.float64)

    # each processor computes its las phis and psis

    if mx == 0:
        phi[:, :, 0] = 0
        psi[:, :, 0] = 1
    else:
        phi[:, :, 0] = beta_local[0]*r[:, :, 0]
        psi[:, :, 0] = -(1./4)*beta_local[0]

    for i in range(1, nx):
        phi[:, :, i] = beta_local[i]*(r[:, :, i] - (1./4)*phi[:, :, i-1])
        psi[:, :, i] = -(1./4)*beta_local[i]*psi[:, :, i-1]

    if mx == npx-1:
        phi[:, :, i] = beta_local[i]*(r[:, :, i] - 2*phi[:, :, i-1])
        psi[:, :, i] = -2*beta_local[i]*psi[:, :, i-1]

    comm.Barrier()

    # each processor posts its last phi and psi
    phi_lasts = np.zeros([nz, ny, npx], dtype=np.float64)
    psi_lasts = np.zeros([nz, ny, npx], dtype=np.float64)

    line_allgather_faces(comm, phi, phi_lasts, 'last')
    line_allgather_faces(comm, psi, psi_lasts, 'last')

    # each processor uses the last phi and psi from the
    # previous processors to compute its u_tilda;
    # the first processor just uses u_0

    u_first = np.zeros([nz, ny], dtype=np.float64)
    u_tilda = np.zeros([nz, ny], dtype=np.float64)
    product_2 = np.zeros([nz, ny], dtype=np.float64)
    product_1 = np.zeros([nz, ny], dtype=np.float64)

    if mx == 0:
        u[:, :, 0] = beta_local[0]*r[:, :, 0]
        u_tilda[:, :] = u[:, :, 0]
        u_first[:, :] = u[:, :, 0]

    comm.Barrier()

    line_bcast(comm, u_first, line_root)

    if rank != line_root:
        u_tilda[...] = 0.0
        product_2[...] = 1.0
        for i in range(mx):
            product_1[...] = 1.0
            for j in range(i+1, mx):
                product_1 *= psi_lasts[:, :, j]
            u_tilda += phi_lasts[:, :, i]*product_1
            product_2 *= psi_lasts[:, :, i]
        u_tilda += u_first*product_2

    comm.Barrier()

    # Now, the entire `u` can be computed:
    for i in range(nx):
        u[:, :, i] = phi[:, :, i] + u_tilda*psi[:, :, i]

    comm.Barrier()

    #############
    # R-L sweep
    #############

    # each processor will need the first `gam` from the next processor:
    gam_firsts = np.zeros(npx, dtype=np.float64)
    line_allgather(comm, gam_local, gam_firsts)
    comm.Barrier()

    if rank == line_last:
        phi[:, :, -1] = 0.0
        psi[:, :, -1] = 1.
    else:
        phi[:, :, -1] = u[:, :, -1]
        psi[:, :, -1] = -gam_firsts[mx+1]

    for i in range(1, nx):
        phi[:, :, -1-i] = u[:, :, -1-i] - gam_local[-1-i+1]*phi[:, :, -1-i+1]
        psi[:, :, -1-i] = -gam_local[-1-i+1]*psi[:, :, -1-i+1]

    comm.Barrier()

    # each processor posts its first phi and psi:
    phi_firsts = np.zeros([nz, ny, npx], dtype=np.float64)
    psi_firsts = np.zeros([nz, ny, npx], dtype=np.float64)

    line_allgather_faces(comm, phi, phi_firsts, 'first')
    line_allgather_faces(comm, psi, psi_firsts, 'first')

    # each processor uses the first phi and psi from the
    # next processors to compute its x_tilda;
    # the last processor just uses x_(n-1)

    x_last = np.zeros([nz, ny], dtype=np.float64)
    x_tilda = np.zeros([nz, ny], dtype=np.float64)

    if rank == line_last:
        x[:, :, -1] = u[:, :, -1]
        x_tilda[:, :] = x[:, :, -1]
        x_last[:, :] = x_tilda[...]

    comm.Barrier()

    line_bcast(comm, x_last, line_last)

    if rank != line_last:
        x_tilda[...] = 0.0
        for i in range(mx+2, line_nprocs):
            product_1[...] = 1.0
            for j in range(mx+1, i):
                product_1 *= psi_firsts[:, :, j]
            x_tilda += phi_firsts[:, :, i]*product_1

        product_2[...] = 1.0
        for i in range(mx+1, line_nprocs):
            product_2 *= psi_firsts[:, :, i]

        x_tilda += phi_firsts[:, :, mx+1] + x_last*product_2

    comm.Barrier()

    for i in range(nx):
        x[:, :, -1-i] = phi[:, :, -1-i] + x_tilda*psi[:, :, -1-i]

    comm.Barrier()
    return x
