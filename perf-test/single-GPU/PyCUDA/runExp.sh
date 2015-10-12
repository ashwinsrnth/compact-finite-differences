
for size in 32 64 256 512
do
    outfile=results/$size.txt
    python run.py $size $size $size > $outfile
done
