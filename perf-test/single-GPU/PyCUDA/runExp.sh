
for solver in globalmem templated
do
    for size in 32 64 128 256 512
    do
        outfile=results/$solver/$size.txt
        echo python run.py $size $size $size $solver
        python run.py $size $size $size $solver > $outfile
    done
done
