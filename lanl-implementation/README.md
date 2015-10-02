# C implementation of `lanl-solver`.

Known issues:

There is still a problem where the following do not give the same result:

mpiexec -n 1 ./time-npts.run 1 1 256 1 1 1
mpiexec -n 16 ./time-npts.run 1 1 256 1 1 16

.. with a high error being reported for the last elements
in each block. This probably means that the error comes from
some pointwise update.

