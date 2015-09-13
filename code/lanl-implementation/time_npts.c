#include <npts.h>
#include <arraytools.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    double *f_global, *r_global;
    double *beta_local, *gam_local, *r_local, *x_local, *f_local;
    double L, dx;
    unsigned int *seeds;
    unsigned int local_seed;
    double t1, t2;
    int local_size, system_size;
    int i, j;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    system_size = atoi(argv[1]);
    local_size = system_size/nprocs;

    beta_local = (double*) malloc(local_size*sizeof(double));
    gam_local = (double*) malloc(local_size*sizeof(double));
    r_local = (double*) malloc(local_size*sizeof(double));
    x_local = (double*) malloc(local_size*sizeof(double));

    /* Generate random RHS elements: */
    if (rank == 0) {
        f_global = (double*) malloc(system_size*sizeof(double));
    }

    f_local = (double*) malloc(system_size*sizeof(double));

    L = (2*3.14159265359);
    dx = L/(system_size-1);
    // create the function and rhs at rank=0
    for (i=0; i<system_size; i++) {
        f_global[i] = sin(i*dx);
    }

    MPI_Scatter(f_global, local_size, MPI_DOUBLE, f_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* f;
        f = fopen("f.txt", "w");
        for (i=0; i<system_size; i++) {
            fprintf(f, "%.12f\n", f_global[i]);
        }
        fclose(f);
    }

    /* Solve the system in parallel */
    precompute_beta_gam(MPI_COMM_WORLD, system_size, beta_local, gam_local);
    MPI_Barrier(MPI_COMM_WORLD);

    t1 = MPI_Wtime();
    for (i=0; i<local_size; i++) {
        for (j=0; j<local_size; j++) {
            nonperiodic_tridiagonal_solver(MPI_COMM_WORLD, beta_local, gam_local,
                f_local, system_size, x_local);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    t2 = MPI_Wtime();

    printf("%f\n", t2-t1);

    if (rank == 0) {
        free(f_global);
    }

    free(beta_local);
    free(gam_local);
    free(r_local);
    free(x_local);
    free(f_local);

    MPI_Finalize();
    return 0;
}
