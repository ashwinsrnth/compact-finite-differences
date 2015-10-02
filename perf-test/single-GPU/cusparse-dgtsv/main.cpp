/*
This script sets up and solves
a "strided batch system", i.e.,
several tridiagonal systems of the following form:

Ay(i) = r(i)

Where A is a tridiagonal coefficient matrix,
and y(i), r(i) are the solution and RHS respectively.

Command-line arguments:
nx - Length of tridiagonal systems
ny, nz - Specifiy the number of systems to solve (nz*ny) 

Outputs:
Time in seconds to solve the problem on GPU (including host-device transfers)
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>


double getRandomDouble()
{
    return ((double)rand()/(double)RAND_MAX);
}


void initRand(double* ary, int size)
{
    /* Initialize ary with random doubles */
    for (int i=0; i<size; i++) {
        ary[i] = getRandomDouble();
    }
}

void initConst(double *ary, double a, int size)
{
    /* Initialize ary with the constant 'a' */
    for (int i=0; i<size; i++) {
        ary[i] = a;
    }
}


void printArray(double* ary, int size) 
{
    for (int i=0; i<size; i++) {
        printf("%f\n", ary[i]);
    }
    printf("\n");
}

void solve_tridiagonal_in_place_reusable(double x[], const size_t N, const double a[], const double b[], const double c[]) {
    size_t in;

    /* Allocate scratch space. */
    double* cprime = (double*)malloc(sizeof(double) * N);

    if (!cprime) {
        /* do something to handle error */
    }

    cprime[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    /* loop from 1 to N - 1 inclusive */
    for (in = 1; in < N; in++) {
        double m = 1.0 / (b[in] - a[in] * cprime[in - 1]);
        cprime[in] = c[in] * m;
        x[in] = (x[in] - a[in] * x[in - 1]) * m;
    }

    /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
    for (in = N - 1; in-- > 0; )
        x[in] = x[in] - cprime[in] * x[in + 1];

    /* free scratch space */
    free(cprime);
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    int nx, ny, nz;
    double *a, *b, *c, *d, *d2;
    double *a_d, *b_d, *c_d, *d_d;
    float milliseconds, total_time;

    cudaEvent_t start, stop;

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    /* Get the size (command line argument) */
    if (argc < 4) {
        printf("Provide nx, ny, nz as a command-line argument!\n");
        return 1;
    }
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);

    total_time = 0;

    /* Generate the three coefficient arrays and a RHS */ 
    a = (double*)malloc(nx*sizeof(double));
    b = (double*)malloc(nx*sizeof(double));
    c = (double*)malloc(nx*sizeof(double));
    d = (double*)malloc(nx*ny*nz*sizeof(double));

    initConst(a, 1./4, nx);
    initConst(b, 1., nx); 
    initConst(c, 1./4, nx);
    initRand(d, nx*ny*nz);
    
    a[0] = 0;
    c[nx-1] = 0;
    a[nx-1] = 2;
    c[0] = 2;

    /* Push the arrays on to the device */
    cudaMalloc((void**)&a_d, nx*sizeof(double));
    cudaMalloc((void**)&b_d, nx*sizeof(double));
    cudaMalloc((void**)&c_d, nx*sizeof(double));
    cudaMalloc((void**)&d_d, nx*ny*nz*sizeof(double));
    cudaMemcpy(a_d, a, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, nx*sizeof(double), cudaMemcpyHostToDevice);

    for (int i=0; i < 10; i++)
    {
        // Call `gtsv` and get the solution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemcpy(d_d, d, nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);        
        cudaEventRecord(start);
        cusparseStatus = cusparseDgtsvStridedBatch(cusparseHandle, nx, a_d, b_d, c_d, d_d, nz*ny, nx);
        cudaEventRecord(stop); 
        cudaMemcpy(d, d_d, nx*ny*nz*sizeof(double), cudaMemcpyDeviceToHost);

        if (cusparseStatus == CUSPARSE_STATUS_ALLOC_FAILED) {
            printf("The resources could not be allocated \n");
        }


        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        printf("%f\n", milliseconds/1000.0);
    }
    printf("%f\n", (total_time/1000.0)/10.0);

    /* Check that the solution is correct */
    // solve_tridiagonal_in_place_reusable(d2, N, a, b, c);
    // printArray(d, N);
    // printArray(d2, N);

    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(d_d);

}
