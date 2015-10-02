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
    double *beta_global, *gamma_global, *r_global, *x_global, *f_global, *d_global, *u_global, *phi, *psi;
    double t1, t2;
    int nx, ny, nz, NX, NY, NZ;
    int npx, npy, npz, mx, my, mz;
    int coords[3];
    int i, j, k, i3d;

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

    /* Set RHS */
    d_global = (double*) malloc(nz*ny*nx*sizeof(double));

    for (i=0; i<nx*ny*nz; i++) {
        d_global[i] = 1.0;
    }

    /* Now every process has the RHS, solve the tridiagonal systems: */
    beta_global =  (double*) malloc(nx*sizeof(double));
    gamma_global = (double*) malloc(nx*sizeof(double));
    
    if (rank == 0) {
        printf("Precomputing coefficients \n");
    }
    MPI_Barrier(comm);
    precompute_beta_gam(comm, NX, NY, NZ, beta_global,\
            gamma_global);
    MPI_Barrier(comm);

    if (rank == 0) {
        printf("Done precomputing. \n");
    }
    u_global = (double*) malloc(nz*ny*nx*sizeof(double));
    phi = (double*) malloc(nz*ny*nx*sizeof(double));
    psi = (double*) malloc(nz*ny*nx*sizeof(double));

    for(i=0; i<nz; i++) {
        for(j=0; j<ny; j++) {
            for (k=0; k<nx; k++) {
                i3d = i*(nx*ny) + j*nx + k;
                u_global[i3d] = 0.0;
            }
        }
    }
    for (i=0; i<20; i++) {
        MPI_Barrier(comm);
        t1 = MPI_Wtime();
        nonperiodic_tridiagonal_solver(comm, NX, NY, NZ, beta_global, gamma_global, d_global, u_global, phi, psi);
        MPI_Barrier(comm);
        t2 = MPI_Wtime();

        if (rank == 0) {
            printf("Time: %f\n", t2-t1);
        }
    }

    free(phi);
    free(psi);
    free(u_global);
    free(beta_global);
    free(gamma_global);
    free(d_global);

    MPI_Finalize();

    return 0;
}
