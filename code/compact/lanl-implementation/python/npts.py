from mpi4py import MPI
import numpy as np
from scipy.linalg import solve_banded

def solve_parallel(comm, rhs_local):
    '''
    Solve the tridiagonal system:
    1/3       1       1/3     .       .       .
      .       1/3     1.      1/3     .       .
      .       .       1/3     1       1/3     .
      .       .       .       .       .       .
      .       .       .       .       .       .
      .       .       .       .       .       .

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

def precompute_beta_gam(comm, system_size):
    # Pre-computes the beta and gam required
    # by the tridiagonal solver
    # The tridiagonal system has:
    # This needs to be done in a pipelined
    # manner, but only once.

    # a[i] = 1./3
    # b[i] = 1.0
    # c[i] = 1./3

    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    local_size = system_size/nprocs

    beta_local = np.zeros(local_size, dtype=np.float64)
    gamma_local = np.zeros(local_size, dtype=np.float64)

    last_beta = np.zeros(1, dtype=np.float64)

    # do a serial hand-off:
    for r in range(nprocs):
        if rank == r:
            if rank == 0:
                beta_local[0] = 1.0
                gamma_local[0] = 0.0
            else:
                comm.Recv([last_beta, 1, MPI.DOUBLE], source=rank-1, tag=10)
                beta_local[0] = 1./(1. - (1./3)*last_beta*(1./3))
                gamma_local[0] = last_beta*(1./3)

            for i in range(1, local_size):
                beta_local[i] = 1./(1. - (1./3)*beta_local[i-1]*(1./3))
                gamma_local[i] = beta_local[i-1]*(1./3)

            if rank != nprocs-1:
                comm.Send([beta_local[-1:], 1, MPI.DOUBLE], dest=rank+1, tag=10)

    return beta_local, gamma_local

def nonperiodic_tridiagonal_solver(comm, beta_local, gam_local, r_local, system_size):
    local_size = beta_local.size
    x_local = np.zeros(local_size, dtype=np.float64)
    u_local = np.zeros(local_size, dtype=np.float64)

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #############
    # L-R sweep
    #############

    assert(system_size%nprocs == 0)

    phi_local = np.zeros(local_size, dtype=np.float64)
    psi_local = np.zeros(local_size, dtype=np.float64)

    # each processor computes its phi and psi
    if rank == 0:
        phi_local[0] = 0
        psi_local[0] = 1
    else:
        phi_local[0] = beta_local[0]*r_local[0]
        psi_local[0] = -(1./3)*beta_local[0]

    for i in range(1, local_size):
        phi_local[i] = beta_local[i]*(r_local[i] - (1./3)*phi_local[i-1])
        psi_local[i] = -(1./3)*beta_local[i]*psi_local[i-1]

    comm.Barrier()

    # each processor posts its last phi and psi
    phi_lasts = np.zeros(nprocs, dtype=np.float64)
    psi_lasts = np.zeros(nprocs, dtype=np.float64)
    comm.Allgather([phi_local[-1:], MPI.DOUBLE], [phi_lasts, MPI.DOUBLE])
    comm.Allgather([psi_local[-1:], MPI.DOUBLE], [psi_lasts, MPI.DOUBLE])

    # each processor uses the last phi and psi from the
    # previous processors to compute its u_tilda;
    # the first processor just uses u_0

    u_first = np.zeros(1, dtype=np.float64)
    if rank == 0:
        u_local[0] = beta_local[0]*r_local[0]
        u_tilda = u_local[0]
        u_first[0] = u_local[0]

    comm.Bcast([u_first, MPI.DOUBLE])

    if rank != 0:
        u_tilda = 0.0
        product_2 = 1.0
        for i in range(rank):
            product_1 = 1.0
            for j in range(i+1, rank):
                product_1 *= psi_lasts[j]
            u_tilda += phi_lasts[i]*product_1
            product_2 *= psi_lasts[i]
        u_tilda += u_first*product_2

    comm.Barrier()

    # Now, the entire `u` can be computed:
    for i in range(local_size):
        u_local[i] = phi_local[i] + u_tilda*psi_local[i]

    comm.Barrier()

    #############
    # R-L sweep
    #############

    # each processor will need the first `gam` from the next processor:
    gam_firsts = np.zeros(nprocs)
    comm.Allgather([gam_local, 1, MPI.DOUBLE], [gam_firsts, MPI.DOUBLE])

    if rank == nprocs-1:
        phi_local[-1] = 0.0
        psi_local[-1] = 1.0
    else:
        phi_local[-1] = u_local[-1]
        psi_local[-1] = -gam_firsts[rank+1]

    for i in range(1, local_size):
        phi_local[-1-i] = u_local[-1-i] - gam_local[-1-i+1]*phi_local[-1-i+1]
        psi_local[-1-i] = -gam_local[-1-i+1]*psi_local[-1-i+1]

    comm.Barrier()

    # each processor posts its first phi and psi:
    phi_firsts = np.zeros(nprocs, dtype=np.float64)
    psi_firsts = np.zeros(nprocs, dtype=np.float64)
    comm.Allgather([phi_local[0:1], MPI.DOUBLE], [phi_firsts, MPI.DOUBLE])
    comm.Allgather([psi_local[0:1], MPI.DOUBLE], [psi_firsts, MPI.DOUBLE])

    # each processor uses the first phi and psi from the
    # next processors to compute its x_tilda;
    # the last processor just uses x_(n-1)

    x_last = np.zeros(1, dtype=np.float64)
    if rank == nprocs-1:
        x_local[-1] = u_local[-1]
        x_tilda = x_local[-1]
        x_last[0] = x_tilda

    comm.Bcast([x_last, MPI.DOUBLE], root=nprocs-1)

    if rank != nprocs-1:
        x_tilda = 0.0
        for i in range(rank+2, nprocs):
            product_1 = 1.0
            for j in range(rank+1, i):
                product_1 *= psi_firsts[j]
            x_tilda += phi_firsts[i]*product_1

        product_2 = 1.0
        for i in range(rank+1, nprocs):
            product_2 *= psi_firsts[i]

        x_tilda += phi_firsts[rank+1] + x_last*product_2

    comm.Barrier()

    for i in range(local_size):
        x_local[-1-i] = phi_local[-1-i] + x_tilda*psi_local[-1-i]

    comm.Barrier()

    return x_local
