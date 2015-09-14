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
    double *f_global, *d_global;
    double *beta_local, *gam_local, *r_local, *x_local, *f_local, *d_local;
    double L, dx;
    unsigned int *seeds;
    unsigned int local_seed;
    double t1, t2;
    int local_size, NX, NY, NZ;
    int i, j;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    NZ = atoi(argv[1]);
    NY = atoi(argv[2]);
    NX = atoi(argv[3]);

    local_size = NX/nprocs;

    beta_local = (double*) malloc(local_size*sizeof(double));
    gam_local = (double*) malloc(local_size*sizeof(double));
    r_local = (double*) malloc(local_size*sizeof(double));
    x_local = (double*) malloc(local_size*sizeof(double));

    /* Generate random RHS elements: */
    if (rank == 0) {
        f_global = (double*) malloc(NX*sizeof(double));
        d_global = (double*) malloc(NX*sizeof(double));
    } 

    f_local = (double*) malloc(NX*sizeof(double));
    d_local = (double*) malloc(NX*sizeof(double));

    L = (2*3.14159265359);
    dx = L/(NX-1);

    if (rank == 0) {
        // create the function and rhs at rank=0
        for (i=0; i<NX; i++) {
            f_global[i] = sin(i*dx);
        }

        for (i=1; i<NX-1; i++) {
            d_global[i] = (3./4)*(f_global[i+1] - f_global[i-1])/dx;            
        }
        d_global[0] = (-5*f_global[0] + 4*f_global[1] + f_global[2])/(2*dx);
        d_global[NX-1] = (-5*f_global[NX-1] + 4*f_global[NX-2] + f_global[NX-3])/(-2*dx);
    }

    MPI_Scatter(d_global, local_size, MPI_DOUBLE, d_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* f;
        f = fopen("d.txt", "w");
        for (i=0; i<NX; i++) {
            fprintf(f, "%.12f\n", d_global[i]);    
        }
        fclose(f);
    }
    
    // Solve the system in parallel 
    precompute_beta_gam(MPI_COMM_WORLD, NX, beta_local, gam_local);
    MPI_Barrier(MPI_COMM_WORLD);

    t1 = MPI_Wtime();
    for (i=0; i<NZ; i++) {
        for (j=0; j<NY; j++) {
            nonperiodic_tridiagonal_solver(MPI_COMM_WORLD, beta_local, gam_local,
                d_local, NX, x_local);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Running with %d processors in one line.\n", nprocs);
        printf("Corresponds to %d processors in total.\n", (NX/NZ)*(NX/NY)*nprocs);
        printf("Full simulation requires %d nodes in total.\n", (NX/NZ)*(NX/NY)*nprocs/16);
        printf("Total size of problem solved: (%d, %d, %d)\n", NZ, NY, NX);
        printf("Time taken: %f\n", t2-t1);
    }

    MPI_Gather(x_local, local_size, MPI_DOUBLE, f_global, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* f;
        f = fopen("x.txt", "w");
        for (i=0; i<NX; i++) {
            fprintf(f, "%.12f\n", f_global[i]);
        }
        fclose(f);
    } 
    
    if (rank == 0) {
        free(f_global);
        free(d_global);
    }

    free(beta_local);
    free(gam_local);
    free(r_local);
    free(x_local);
    free(f_local);
    free(d_local);

    MPI_Finalize();
    return 0;
}
