#include <npts.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <time.h>
#include <options.h>

void get_line_info(MPI_Comm comm, int *line_root, int *line_processes) {
    /* Get the root process for the x-direction
     * along this line and the ranks of processes in the x-direction.
     *
     * comm: An instance of a Cartcomm
     * line_root: The root process of this line
     * line_processes: An int array sized "npx" that will contain
     *                  the ranks of processes along this line
     */
    int rank;
    int mz, my, mx;
    int npz, npy, npx;
    int i;
    int dims[3], periods[3], coords[3];

    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];

    // get the line root:
    *line_root = (rank/npx)*npx;

    // get the line processes:
    for (i=0; i<npx; i++) {
        line_processes[i] = *line_root+i;
    }
}

void line_subarray(MPI_Comm comm, int *shape, int subarray_length, MPI_Datatype *subarray, int *lengths, int *displacements) {
    /*
     * Create a subarray for orthogonal to this line.
     * If each block in this line is sized [nz, ny, nx],
     * then return a subarray sized [nz, ny, subarray_length].
     * For the case of subarray_length = 1 for instance, this
     * is useful for getting the left and right "faces" of a
     * block.
     */
    int nprocs, rank;
    int mz, my, mx;
    int npz, npy, npx;
    int i;
    int dims[3], periods[3], coords[3];
    MPI_Datatype subarray_aux;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    for (i=0; i<nprocs; i++) {
        lengths[i] = 0;
        displacements[i] = 0;
    }

    for (i=0; i<npx; i++) {
        lengths[line_processes[i]] = subarray_length;
        displacements[line_processes[i]] = i*subarray_length;
    }

    int sizes[3] = {shape[0], shape[1], npx};
    int subsizes[3] = {shape[0], shape[1], subarray_length};
    int starts[3] = {0, 0, displacements[rank]};

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_aux);
    MPI_Type_create_resized(subarray_aux, 0, 8, subarray);
    MPI_Type_commit(subarray);
}

void line_bcast(MPI_Comm comm, double *buf, int count, int root) {
    /*
     * Broadcast from the rank "root" to all other
     * processes in this line.
     */
    int rank;
    int mz, my, mx;
    int npz, npy, npx;
    int i;
    int dims[3], periods[3], coords[3];

    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];

    int line_root;
    int line_processes[npx];
    MPI_Request message;

    get_line_info(comm, &line_root, line_processes);

    if (rank == root) {
        for (i=0; i<npx; i++) {
            if (line_processes[i] != root) {
                MPI_Isend(buf, count, MPI_DOUBLE, line_processes[i], line_processes[i]*10, comm, &message);
            }
        }
    }

    if (rank != root) {
        MPI_Irecv(buf, count, MPI_DOUBLE, root, rank*10, comm, &message);
        MPI_Wait(&message, MPI_STATUS_IGNORE);
    }

}

void line_allgather_faces(MPI_Comm comm, double *x, int *shape, double *x_faces, int face) {
    /*
    Gather the left or right faces of a block from each
    process into the line_root and the broadcast it.

    x: A logically 3-D block shaped [nz, ny, nx]
    shape: The shape of the block

    face: 0 - left [:, :, 0]
          1 - right [:, :, -1]
    */

    int rank, nprocs;
    int mz, my, mx;
    int npz, npy, npx;
    int nz, ny, nx;
    int i, j, k, i3d, i2d;
    int dims[3], periods[3], coords[3];

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];
    nz = shape[0];
    ny = shape[1];
    nx = shape[2];

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    int lengths[nprocs];
    int displacements[nprocs];
    MPI_Datatype subarray;

    line_subarray(comm, shape, 1, &subarray, lengths, displacements);

    double *x_face;
    x_face = (double*) malloc((nz*ny)*sizeof(double));

    // left:
    if (face == 0) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i2d = i*ny + j;
                i3d = i*(nx*ny) + j*nx + 0;
                x_face[i2d] = x[i3d];
            }
        }
    }

    else if (face == 1) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i2d = i*ny + j;
                i3d = i*(nx*ny) + j*nx + nx-1;
                x_face[i2d] = x[i3d];
            }
        }
    }

    else {
        printf("Error: specify a valid face flag.");
    }

    MPI_Barrier(comm);

    MPI_Gatherv(x_face, nz*ny, MPI_DOUBLE, x_faces, lengths, displacements, subarray, line_root, comm);
    line_bcast(comm, x_faces, nz*ny*npx, line_root);

    free(x_face);
    MPI_Type_free(&subarray);
}

void line_allgather(MPI_Comm comm, double *x, double *x_line) {
    /*
    Gather the left or right faces from the blocks in each process
    process into the line_root and the broadcast it.

    x: A logically 3-D block shaped [nz, ny, nx]
    x_line: A logically 3-D block shaped [nz, ny, npx]
            that contains either the left or the right faces
            from each block "x" in this line.
    face: 0 - left
          1 - right
    */

    int rank, nprocs;
    int mz, my, mx;
    int npz, npy, npx;
    int nz, ny, nx;
    int i, j, k, i3d, i2d;
    int dims[3], periods[3], coords[3];

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    int lengths[nprocs];
    int displacements[nprocs];

    for (i=0; i<nprocs; i++) {
        lengths[i] = 0;
        displacements[i] = 0;
    }

    for (i=0; i<npx; i++) {
        lengths[line_processes[i]] = 1;
        displacements[line_processes[i]] = i;
    }

    MPI_Gatherv(x, 1, MPI_DOUBLE, x_line, lengths, displacements, MPI_DOUBLE, line_root, comm);
    line_bcast(comm, x_line, npx, line_root);
}

void nonperiodic_tridiagonal_solver(const MPI_Comm comm, const int NX, const int NY, const int NZ, double* beta_global, \
    double* gam_global, double* r_global, double* u_global, double* phi, double* psi){
    /* Solve the tridiagonal system in parallel with size [NZ, NY, NX]
     * Inputs are named by PETSc convention ("global" just refers to arrays without neighbouring process information)
     * beta, gam: Precomputed coefficents (sized nx)
     * r: Right hand side (sized nx*ny*nz)
     * u: Space for solution (sized nx*ny*nx)
     * phi, psi: Scratch buffers (sized nx*ny*nz)
     */

    int rank, nprocs;
    int mz, my, mx;
    int npz, npy, npx;
    int nz, ny, nx;
    int i, j, ii, jj, k, m, n, i3d, i2d;
    int dims[3], periods[3], coords[3];
    double t1, t2;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];
    nz = NZ/npz;
    ny = NY/npy;
    nx = NX/npx;
    int shape[3] = {nz, ny, nx};

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    /* LR-sweep */

    t1 = MPI_Wtime();
    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                phi[i3d] = 0;
                psi[i3d] = 0;
            }
        }
    }
    
    MPI_Barrier(comm);

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            i3d = i*(nx*ny) + j*nx + 0;
            if (mx == 0) {
                phi[i3d] = 0;
                psi[i3d] = 1;
            }
            else {
                phi[i3d] = beta_global[0]*r_global[i3d];
                psi[i3d] = -(1./4)*beta_global[0];
            }

            for (k=1; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                phi[i3d] = beta_global[k]*(r_global[i3d] - (1./4)*phi[i3d-1]);
                psi[i3d] = -(1./4)*beta_global[k]*psi[i3d-1];
            }

            if (mx == npx-1) {
                i3d = i*(nx*ny) + j*nx + nx-1;
                phi[i3d] = beta_global[nx-1]*(r_global[i3d] - 2*phi[i3d-1]);
                psi[i3d] = -2*beta_global[nx-1]*psi[i3d-1];
            }
        }
    }

    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("LR sweep - computing phi and psi: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    double *phi_faces, *psi_faces;
    phi_faces = (double*) malloc(nz*ny*npx*sizeof(double));
    psi_faces = (double*) malloc(nz*ny*npx*sizeof(double));

    line_allgather_faces(comm, phi, shape, phi_faces, 1);
    line_allgather_faces(comm, psi, shape, psi_faces, 1);

    MPI_Barrier(comm);
    
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("LR sweep - gathering phi and psi faces: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    double *u_tilda, *u0;
    u_tilda = (double*) malloc(nz*ny*sizeof(double));
    u0 = (double*) malloc(nz*ny*sizeof(double));
    double product_1, product_2;

    if (rank == line_root) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i3d = i*(nx*ny) + j*nx + 0;
                i2d = i*ny + j;
                u_global[i3d] = beta_global[0]*r_global[i3d];
                u_tilda[i2d] = u_global[i3d];
                u0[i2d] = u_global[i3d];
            }
        }
    }

    MPI_Barrier(comm);
    line_bcast(comm, u0, nz*ny, line_root);
    MPI_Barrier(comm);

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            if (rank != line_root) {
                i2d = i*ny + j;
                u_tilda[i2d] = 0.0;
                product_2 = 1.0;
                for (ii=0; ii<mx; ii++) {
                    product_1 = 1.0;
                    for (jj=ii+1; jj<mx; jj++) {
                        i3d = i*(npx*ny) + j*npx + jj;
                        product_1 *= psi_faces[i3d];
                    }
                    i3d = i*(npx*ny) + j*npx + ii;
                    u_tilda[i2d] += phi_faces[i3d]*product_1;
                    product_2 *= psi_faces[i3d];
                }
                u_tilda[i2d] += u0[i2d]*product_2;
            }
        }
    }
    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("LR sweep - computing u_tilda: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                i2d = i*ny + j;
                u_global[i3d] = phi[i3d] + u_tilda[i2d]*psi[i3d];
            }
        }
    }

    MPI_Barrier(comm);
    
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("LR sweep - computing u_globali: %0.10f\n", t2-t1);
    }

    /* R-L sweep */

    double *gam_firsts;
    int line_last;

    t1 = MPI_Wtime();

    gam_firsts = (double *) malloc(npx*sizeof(double));
    line_allgather(comm, &gam_global[0], gam_firsts);
    MPI_Barrier(comm);
    line_last = line_root + npx-1;
    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            i3d = i*(nx*ny) + j*nx + nx-1;
            if (rank == line_last) {
                phi[i3d] = 0.0;
                psi[i3d] = 1.0;
            }

            else {
                phi[i3d] = u_global[i3d];
                psi[i3d] = -gam_firsts[mx+1];
            }

            for (k=1; k<nx; k++) {
                phi[i3d-k] = u_global[i3d-k] - gam_global[nx-1-k+1]*phi[i3d-k+1];
                psi[i3d-k] = -gam_global[nx-1-k+1]*psi[i3d-k+1];
            }
        }
    }

    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("R-L sweep - computing phi and psi: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    line_allgather_faces(comm, phi, shape, phi_faces, 0);
    line_allgather_faces(comm, psi, shape, psi_faces, 0);
    
    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("R-L sweep - gathering phi and psi faces: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    if (rank == line_last) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i3d = i*(nx*ny) + j*nx + nx-1;
                i2d = i*ny + j;
                u_tilda[i2d] = u_global[i3d];
                u0[i2d] = u_tilda[i2d];
            }
        }
    }

    MPI_Barrier(comm); 

    line_bcast(comm, u0, nz*ny, line_last);
    
    MPI_Barrier(comm);

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            if (rank != line_last) {
                i2d = i*ny + j;
                u_tilda[i2d] = 0.0;
                
                for (ii=mx+2; ii<npx; ii++) {
                    product_1 = 0.0;
                    for (jj=mx+1; jj<ii; jj++) {
                        i3d = i*(npx*ny) + j*npx + jj;
                        product_1 *= psi_faces[i3d];
                    }
                    i3d = i*(npx*ny) + j*npx + ii;
                    u_tilda[i2d] += phi_faces[i3d]*product_1;
                }
                product_2 = 1.0;
                for (ii=mx+1; ii<npx; ii++) {
                    i3d = i*(npx*ny) + j*npx + ii;
                    product_2 *= psi_faces[i3d];
                }
                
                i3d = i*(npx*ny) + j*npx + mx+1;
                u_tilda[i2d] += phi_faces[i3d] + u0[i2d]*product_2;
            }
        }
    }
    
    MPI_Barrier(comm);  
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("R-L sweep - computing x_tilda: %0.10f\n", t2-t1);
    }

    t1 = MPI_Wtime();

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + nx-1;
                i2d = i*ny + j;
                u_global[i3d-k] = phi[i3d-k] + u_tilda[i2d]*psi[i3d-k];
            }
        }
    }
    
    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (rank == 0 && PRINT_TIMINGS) {
        printf("R-L sweep - computing x_global: %0.10f\n", t2-t1);
    }

    free(gam_firsts);
    free(phi_faces);
    free(psi_faces);
    free(u_tilda);
    free(u0);
}



void precompute_beta_gam(MPI_Comm comm, int NX, int NY, int NZ, double* beta_global,
    double* gamma_global)
{
    int rank, nprocs;
    int coords[3], dims[3], periods[3];
    int i, j, k, i2d, i3d;
    int nz, ny, nx;
    int npz, npy, npx, mz, my, mx;
    int r;
    double last_beta;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Cart_coords(comm, rank, 3, coords);
    MPI_Cart_get(comm, 3, dims, periods, coords);

    npz = dims[0];
    npy = dims[1];
    npx = dims[2];
    mz = coords[0];
    my = coords[1];
    mx = coords[2];
    nx = NX/npx;
    ny = NY/npy;
    nz = NZ/npz;

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    for (i=0; i<nx; i++) {
        beta_global[i] = 0;
        gamma_global[i] = 0;
    }

    last_beta = 0;

    /* Do the serial handoff */
    for (r=line_root; r<=line_root+npx-1; r++) {
        if (rank == r) {
            if (rank == line_root) {
                beta_global[0] = 1.0;
                gamma_global[0] = 0.0;
            }
            else {
                MPI_Recv(&last_beta, 1, MPI_DOUBLE, rank-1, 11, comm, MPI_STATUS_IGNORE);
                beta_global[0] = 1./(1. - (1./4)*last_beta*(1./4));
                gamma_global[0] = last_beta*(1./4);
            }

            for (i=1; i<nx; i++) {
                if (rank == line_root && i == 1) {
                    gamma_global[i] = beta_global[i-1]*2;
                }
                else {
                    gamma_global[i] = beta_global[i-1]*(1./4);
                }

                if (rank == line_root+npx-1 && i == nx-1) {
                    beta_global[i] = 1./(1. - (2.0)*beta_global[i-1]*(1./4));
                }
                else if (rank == line_root && i == 1) {
                    beta_global[i] = 1./(1. - (2.0)*beta_global[i-1]*(1./4));
                }
                else {
                    beta_global[i] = 1./(1. - (1./4)*beta_global[i-1]*(1./4));
                }
            }

            if (rank != line_root+npx-1) {
                MPI_Send(&beta_global[nx-1], 1, MPI_DOUBLE, rank+1, 11, comm);
            }
        }
    }
}
