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

#define PI 3.14159265359

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm;
    int rank, nprocs;
    double *f_line, *d_line;
    double *beta_local, *gam_local, *r_local, *x_local, *f_local, *d_local;
    double t1, t2;
    int nx, ny, nz, NX, NY, NZ;
    int npx, npy, npz, mx, my, mz;
    int coords[3];
    int i, j, k;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    NZ = atoi(argv[1]);
    NY = atoi(argv[2]);
    NX = atoi(argv[3]);
    npz = atoi(argv[4]);
    npy = atoi(argv[5]);
    npx = atoi(argv[6]);

    nx = NX/npz;
    ny = NY/npy;
    nz = NZ/npz;

    assert(nprocs==npz*npy*npz);

    /* Create communicator */
    const int dims[3] = {npz, npy, npx};
    const int periods[3] = {0, 0, 0};

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 3, coords);

    mx = coords[0];
    my = coords[1];
    mz = coords[2];

    /*
    Create a subarray type
    */
    int sizes[3] = {NZ, NY, NX};
    int subsizes[3] = {nz, ny, nx};
    int starts[3] = {mz*npz, my*npy, mx*npx};
    MPI_Datatype subarray_aux, subarray;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_aux);
    MPI_Type_create_resized(subarray_aux, 0, 8, &subarray);
    MPI_Type_commit(&subarray);
    MPI_Type_free(&subarray);


    /*
    Create the RHS
    */

    double *f_full, *d_full;
    double dx = 2.0*PI/(NX-1);
    int i3d, i_1, i_N;

    if (rank == 0) {
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
                d_full[i_N] = (-5*f_full[i_N-1] + 4*f_full[i_N-2] + f_full[i_1-3])/(-2*dx);
            }
        }

    }

    /* Scatter the RHS */
    double *d_global;
    d_global = (double*) malloc(nz*ny*nx*sizeof(double));

    MPI_Scatter(d_full, 1, subarray, d_global, nz*ny*nx, MPI_DOUBLE, 0, comm);

    /* Now every process has the RHS, solve the tridiagonal systems: */



    if (rank == 0) {
        free(f_full);
        free(d_full);
    }

    free(d_global);

    MPI_Finalize();

    return 0;
}
