#include <npts.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <time.h>

void get_line_info(MPI_Comm comm, int *line_root, int *line_processes) {
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
    Gather the left or right faces from each
    process into the line_root and the broadcast it.

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
    Gather the left or right faces from each
    process into the line_root and the broadcast it.

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

void nonperiodic_tridiagonal_solver(MPI_Comm comm, int NX, int NY, int NZ, double* beta_global, \
    double* gam_global, double* r_global, double* x_global, double* u_global){

    int rank, nprocs;
    int mz, my, mx;
    int npz, npy, npx;
    int nz, ny, nx;
    int i, j, ii, jj, k, m, n, i3d, i2d;
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
    nz = NZ/npz;
    ny = NY/npy;
    nx = NX/npx;
    int shape[3] = {nz, ny, nx};

    int line_root;
    int line_processes[npx];

    get_line_info(comm, &line_root, line_processes);

    /* LR-sweep */
    double *phi, *psi;
    phi = (double*) malloc(nz*ny*nx*sizeof(double));
    psi = (double*) malloc(nz*ny*nx*sizeof(double));

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                phi[i3d] = 0;
                psi[i3d] = 0;
            }
        }
    }

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

    i3d = (nz-1)*(nx*ny) + (ny-1)*nx + (nx-1);

    double *phi_lasts, *psi_lasts;
    phi_lasts = (double*) malloc(nz*ny*npx*sizeof(double));
    psi_lasts = (double*) malloc(nz*ny*npx*sizeof(double));

    line_allgather_faces(comm, phi, shape, phi_lasts, 1);
    line_allgather_faces(comm, psi, shape, psi_lasts, 1);

    double *u_tilda, *u_first;
    u_tilda = (double*) malloc(nz*ny*sizeof(double));
    u_first = (double*) malloc(nz*ny*sizeof(double));
    double product_1, product_2;

    if (mx == 0) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i3d = i*(nx*ny) + j*nx + 0;
                i2d = i*ny + j;
                u_global[i3d] = beta_global[0]*r_global[i3d];
                u_tilda[i2d] = u_global[i3d];
                u_first[i2d] = u_global[i3d];
            }
        }
    }

    MPI_Barrier(comm);
    line_bcast(comm, u_first, nz*ny, line_root);


    if (rank != line_root) {
        for (i=0; i<nz; i++) {
            for (j=0; j<ny; j++) {
                i2d = i*ny + j;
                u_tilda[i2d] = 0.0;
                product_2 = 1.0;
                for (ii=0; ii<mx; ii++) {
                    product_1 = 1.0;
                    for (jj=ii+1; jj<mx; jj++) {
                        i3d = i*(mx*ny) + j*mx + jj;
                        product_1 *= psi_lasts[i3d];
                    }
                    i3d = i*(npx*ny) + j*npx + ii;
                    u_tilda[i2d] += phi_lasts[i3d]*product_1;
                    product_2 *= psi_lasts[i3d];
                }
                u_tilda[i2d] += u_first[i2d]*product_2;
            }
        }
    }

    for (i=0; i<nz; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                i2d = i*(ny) + j;
                u_global[i3d] = phi[i3d] + u_tilda[i2d]*psi[i3d];
            }
        }
    }

    i3d = (nz-1)*(ny*nx) + (ny-1)*nx + nx-1;
    printf("%f\n", u_global[i3d]);


    free(phi);
    free(psi);
    free(phi_lasts);
    free(psi_lasts);
    free(u_tilda);
    free(u_first);


    // double *phi_local, *psi_local;
    // double *phi_lasts, *psi_lasts, *gam_firsts, *phi_firsts, *psi_firsts;
    // double u_tilda, x_tilda, u_first, x_last;
    // double product_1, product_2;
    // int rank, nprocs;
    // int local_size;
    // int i, j;
    // double tstart, tend, t1, t2;
    //
    // tstart = MPI_Wtime();
    //
    // t1 = MPI_Wtime();
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &nprocs);
    // local_size = system_size/nprocs;
    //
    // if (system_size%nprocs != 0) {
    //     printf("Error: system_size (%zd) not a multiple of nprocs (%d).\n", system_size, nprocs);
    //     return;
    // }
    //
    // u_local = (double*) malloc(local_size*sizeof(double));
    // phi_local = (double*) malloc(local_size*sizeof(double));
    // psi_local = (double*) malloc(local_size*sizeof(double));
    // phi_lasts = (double*) malloc(nprocs*sizeof(double));
    // psi_lasts = (double*) malloc(nprocs*sizeof(double));
    // phi_firsts = (double*) malloc(nprocs*sizeof(double));
    // psi_firsts = (double*) malloc(nprocs*sizeof(double));
    // gam_firsts = (double*) malloc(nprocs*sizeof(double));
    //
    // initZeros(u_local, local_size);
    //
    // /* --------- */
    // /* L-R sweep */
    // /* --------- */
    //
    // if (rank == 0) {
    //     phi_local[0] = 0.0;
    //     psi_local*,0] = 1.0;
    // }
    //
    // else {
    //     phi_local[0] = beta_local[0]*r_local[0];
    //     psi_local[0] = -(1./4)*beta_local[0];
    // }
    //
    // for(i=1; i<local_size; i++) {
    //     phi_local[i] = beta_local[i]*(r_local[i] - (1./4)*phi_local[i-1]);
    //     psi_local[i] = -(1./4)*beta_local[i]*psi_local[i-1];
    // }
    //
    // if (rank == nprocs-1) {
    //     phi_local[local_size-1] = beta_local[local_size-1]*(r_local[local_size-1] - 2*phi_local[local_size-2]);
    //     psi_local[local_size-1] = -2*beta_local[local_size-1]*psi_local[local_size-2];
    // }
    //
    // MPI_Barrier(comm);
    //
    // MPI_Allgather(&phi_local[local_size-1], 1, MPI_DOUBLE, phi_lasts,
    //     1, MPI_DOUBLE, comm);
    //
    // MPI_Allgather(&psi_local[local_size-1], 1, MPI_DOUBLE, psi_lasts,
    //     1, MPI_DOUBLE, comm);
    //
    // u_first = 0;
    // if (rank == 0) {
    //     u_local[0] = beta_local[0]*r_local[0];
    //     u_tilda = u_local[0];
    //     u_first = u_local[0];
    // }
    //
    // MPI_Bcast(&u_first, 1, MPI_DOUBLE, 0, comm);
    //
    // if (rank != 0) {
    //     u_tilda = 0.0;
    //     product_2 = 1.0;
    //     for (i=0; i<rank; i++) {
    //         product_1 = 1.0;
    //         for (j=i+1; j<rank; j++) {
    //             product_1 *= psi_lasts[j];
    //         }
    //         u_tilda += phi_lasts[i]*product_1;
    //         product_2 *= psi_lasts[i];
    //     }
    //     u_tilda += u_first*product_2;
    // }
    //
    // MPI_Barrier(comm);
    //
    // for (i=0; i<local_size; i++) {
    //     u_local[i] = phi_local[i] + u_tilda*psi_local[i];
    // }
    //
    // MPI_Barrier(comm);
    //
    // /* --------- */
    // /* R-L sweep */
    // /* --------- */
    //
    // initZeros(gam_firsts, nprocs);
    // MPI_Allgather(gam_local, 1, MPI_DOUBLE, gam_firsts, 1, MPI_DOUBLE, comm);
    //
    // if (rank == nprocs-1) {
    //     phi_local[local_size-1] = 0.0;
    //     psi_local[local_size-1] = 1.0;
    // }
    //
    // else {
    //     phi_local[local_size-1] = u_local[local_size-1];
    //     psi_local[local_size-1] = -gam_firsts[rank+1];
    // }
    //
    // for (i=1; i<local_size; i++) {
    //     phi_local[local_size-1-i] = u_local[local_size-1-i] -
    //         gam_local[local_size-1-i+1]*phi_local[local_size-1-i+1];
    //     psi_local[local_size-1-i] = -gam_local[local_size-1-i+1]*psi_local[local_size-1-i+1];
    // }
    //
    // MPI_Barrier(comm);
    //
    // MPI_Allgather(phi_local, 1, MPI_DOUBLE, phi_firsts, 1, MPI_DOUBLE, comm);
    // MPI_Allgather(psi_local, 1, MPI_DOUBLE, psi_firsts, 1, MPI_DOUBLE, comm);
    //
    // x_last = 0.0;
    // if (rank == nprocs-1) {
    //     x_local[local_size-1] = u_local[local_size-1];
    //     x_tilda = x_local[local_size-1];
    //     x_last = x_tilda;
    // }
    //
    // MPI_Bcast(&x_last, 1, MPI_DOUBLE, nprocs-1, comm);
    //
    // if (rank != nprocs-1) {
    //     x_tilda = 0.0;
    //     for (i=rank+2; i<nprocs; i++) {
    //         product_1 = 1.0;
    //         for (j=rank+1; j<i; j++) {
    //             product_1 *= psi_firsts[j];
    //         }
    //         x_tilda += phi_firsts[i]*product_1;
    //     }
    //
    //     product_2 = 1.0;
    //     for (i=rank+1; i<nprocs; i++) {
    //         product_2 *= psi_firsts[i];
    //     }
    //
    //     x_tilda += phi_firsts[rank+1] + x_last*product_2;
    // }
    //
    // MPI_Barrier(comm);
    //
    // for (i=0; i<local_size; i++) {
    //     x_local[local_size-1-i] = phi_local[local_size-1-i] +
    //         x_tilda*psi_local[local_size-1-i];
    // }
    //
    // MPI_Barrier(comm);
    //
    // free(phi_lasts);
    // free(psi_lasts);
    // free(phi_local);
    // free(psi_local);
    // free(u_local);
    //
    // MPI_Barrier(comm);
    // tend = MPI_Wtime();
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
