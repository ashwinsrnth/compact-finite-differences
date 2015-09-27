#pragma once

#include <stdio.h>
#include <mpi.h>

void nonperiodic_tridiagonal_solver(const MPI_Comm comm, const int NX, const int NY, const int NZ, double* beta_global, \
    double* gam_global, double* r_global, double* u_global, double* phi, double* psi);
void precompute_beta_gam(MPI_Comm comm, int NX, int NY, int NZ, double* beta_global, \
    double* gamma_global);
void get_line_info(MPI_Comm comm, int *line_root, int *line_processes);
void line_subarray(MPI_Comm comm, int *shape, int subarray_length, MPI_Datatype *subarray, int *lengths, int *displacements);
void line_bcast(MPI_Comm comm, double *buf, int count, int root);
void line_allgather_faces(MPI_Comm comm, double *x, int *shape, double *x_faces, int face);
void line_allgather(MPI_Comm comm, double *x, double *x_line);
