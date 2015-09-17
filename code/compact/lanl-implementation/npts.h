#pragma once

#include <stdio.h>
#include <mpi.h>

void nonperiodic_tridiagonal_solver(MPI_Comm comm, double* beta_local, double* gam_local, \
    double* r_local, size_t system_size,  double* x_local);
void precompute_beta_gam(MPI_Comm comm, int NX, int NY, int NZ, double* beta_local, \
    double* gamma_local);
void get_line_info(MPI_Comm comm, int *line_root, int *line_processes);
void line_subarray(MPI_Comm comm, int *shape, int subarray_length, MPI_Datatype *subarray, int *lengths, int *displacements);
void line_bcast(MPI_Comm comm, double *buf, int count, int root);
void line_allgather_faces(MPI_Comm comm, double *x, int *shape, double *x_faces, int face);
void line_allgather(MPI_Comm comm, double *x, double *x_line);
void nonperiodic_tridiagonal_solver(MPI_Comm comm, double* beta_local, \
    double* gam_local, double* r_local, size_t system_size, double* x_local);
