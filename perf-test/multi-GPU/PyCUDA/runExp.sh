#!/bin/bash
MPIFLAGS='--mca pml ob1 --mca btl_openib_want_cuda_gdr 0 --mca mpi_warn_on_fork 0'

for solver in globalmem templated
do
    for local_size in 32 64 128 256 512
    do
        for x_procs in 2 3 4
        do
            total_size=$((local_size*x_procs))
            total_procs=$((x_procs*x_procs*x_procs))
            outfile=results/$solver/$total_size-$total_procs.txt
            echo mpiexec $MPIFLAGS -n $total_procs python run.py $local_size $local_size $local_size $x_procs $x_procs $x_procs $solver
            mpiexec $MPIFLAGS -n $total_procs python run.py $local_size $local_size $local_size $x_procs $x_procs $x_procs $solver > $outfile
        done
    done
done
