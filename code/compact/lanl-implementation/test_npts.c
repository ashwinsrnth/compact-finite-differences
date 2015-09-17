#include <npts.h>
#include <arraytools.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#define PI 3.141592653589793238462643383

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm;
    int rank, nprocs;
    double *f_line, *d_line;
    double *beta_global, *gamma_global, *r_global, *x_global, *f_global, *d_global, *u_global;
    double t1, t2;
    int nx, ny, nz, NX, NY, NZ;
    int npx, npy, npz, mx, my, mz;
    int coords[3];
    int i, j, k;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    NZ = atoi(argv[1]);
    NY = atoi(argv[2]);
    NX = atoi(argv[3]);
    npz = atoi(argv[4]);
    npy = atoi(argv[5]);
    npx = atoi(argv[6]);

    nx = NX/npx;
    ny = NY/npy;
    nz = NZ/npz;

    assert(nprocs==npx*npy*npz);

    /* Create communicator */
    const int dims[3] = {npz, npy, npx};
    const int periods[3] = {0, 0, 0};

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords);

    mz = coords[0];
    my = coords[1];
    mx = coords[2];

    /*
    Create a subarray type
    */
    int sizes[3] = {NZ, NY, NX};
    int subsizes[3] = {nz, ny, nx};
    int starts[3] = {mz*nz, my*ny, mx*nx};
    MPI_Datatype subarray_aux, subarray;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_aux);
    MPI_Type_create_resized(subarray_aux, 0, 8, &subarray);
    MPI_Type_commit(&subarray);

    /*
    Create the RHS
    */

    double *f_full, *d_full;
    double dx = 2.0*PI/(NX-1);
    int i3d, i_1, i_N;

    f_full = (double*) malloc(NZ*NY*NX*sizeof(double));
    d_full = (double*) malloc(NZ*NY*NX*sizeof(double));

    for(i=0; i<NZ; i++) {
        for(j=0; j<NY; j++) {
            for (k=0; k < NX; k++) {
                i3d = i*(NX*NY) + j*NX + k;
                f_full[i3d] = sin(k*dx);
            }
        }
    }

    for(i=0; i<NZ; i++) {
        for(j=0; j<NY; j++) {
            for (k=1; k<NX-1; k++) {
                i3d = i*(NX*NY) + j*NX + k;
                d_full[i3d] = (3./4)*(f_full[i3d+1] - f_full[i3d-1])/dx;
             }
            i_1 = i*(NX*NY) + j*NX + 0;
            i_N = i*(NX*NY) + j*NX + NX-1;
            d_full[i_1] = (-5*f_full[i_1] + 4*f_full[i_1+1] + f_full[i_1+2])/(2*dx);
            d_full[i_N] = (-5*f_full[i_N] + 4*f_full[i_N-1] + f_full[i_N-2])/(-2*dx);
        }
    }

    /* Scatter the RHS */
    d_global = (double*) malloc(nz*ny*nx*sizeof(double));
    int lengths[nprocs];

    for (i=0; i<nprocs; i++) {
        lengths[i] = 1;
    }

    /* Everyone computes a start index: */
    int start_index = mz*nz*(NX*NY) + my*ny*(NX) + mx*nx;
    int displacements[nprocs];
    MPI_Gather(&start_index, 1, MPI_INT, displacements, 1, MPI_INT, 0, comm);

    MPI_Scatterv(d_full, lengths, displacements, subarray, d_global, nz*ny*nx, MPI_DOUBLE, 0, comm);

    /* Now every process has the RHS, solve the tridiagonal systems: */
    beta_global =  (double*) malloc(nx*sizeof(double));
    gamma_global = (double*) malloc(nx*sizeof(double));
    precompute_beta_gam(comm, NX, NY, NZ, beta_global,\
        gamma_global);


    x_global = (double*) malloc(nz*ny*nx*sizeof(double));
    u_global = (double*) malloc(nz*ny*nx*sizeof(double));

    for(i=0; i<nz; i++) {
        for(j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                u_global[i3d] = 0.0;
                x_global[i3d] = 0.0;
            }
        }
    }

    nonperiodic_tridiagonal_solver(comm, NX, NY, NZ, beta_global, gamma_global, d_global, x_global, u_global);

    MPI_Barrier(comm);

    MPI_Gatherv(x_global, ny*nx*nz, MPI_DOUBLE, d_full, lengths, displacements, subarray, 0, comm);
    MPI_Barrier(comm);
    MPI_Type_free(&subarray);


    if (rank == 0) {
        double err = 0;
        double x = 0;
        for(i=0; i<NZ; i++) {
            for(j=0; j<NY; j++) {
                for (k=0; k<NX; k++) {
                    i3d = i*(NX*NY) + j*NX + k;
                    x = k*dx;
                    err += fabs(cos(x) - d_full[i3d]);
                }
            }
        }
        printf("Average absolute error: %0.10f\n", err/(NZ*NX*NY));
    }

    if (rank == 0) {
        free(f_full);
        free(d_full);
    }

    free(x_global);
    free(u_global);
    free(beta_global);
    free(gamma_global);
    free(d_global);

    MPI_Finalize();

    return 0;
}
