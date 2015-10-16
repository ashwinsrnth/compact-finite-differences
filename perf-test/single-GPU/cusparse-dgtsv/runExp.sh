make run

for size in 32 64 128 256 512
do
    outfile=results/$size.txt
    ./run $size $size $size > $outfile
done
