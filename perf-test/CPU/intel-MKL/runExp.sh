make run

rm -f results/results.txt
touch results/results.txt
outfile=results/results.txt
for size in 32 64 128 256 512
do
    ./run $size $size $size >> $outfile
done
