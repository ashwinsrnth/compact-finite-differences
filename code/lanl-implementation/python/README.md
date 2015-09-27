# npts

`npts` (Non Periodic Tridiagonal Solver) is a solver that implements
the method outlined in the paper
"Adapting the CFDNS Compressible Navier-Stokes Solver to the Roadrunner
Hybrid Supercomputer"

 `npts` provides the function `parallel_solve` to solve tridiagonal systems
 of the form:

 1/3       1       1/3     .       .       .
   .       1/3     1.      1/3     .       .
   .       .       1/3     1       1/3     .
   .       .       .       .       .       .
   .       .       .       .       .       .
   .       .       .       .       .       .

in parallel.
