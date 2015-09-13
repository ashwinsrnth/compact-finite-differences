#pragma once

#include <stdio.h>
#include <mpi.h>

void nonperiodic_tridiagonal_solver(MPI_Comm comm, double* beta_local, double* gam_local,
    double* r_local, size_t system_size,  double* x_local);
void precompute_beta_gam(MPI_Comm comm, size_t system_size,
    double* beta_local, double* gam_local);
