
time_npts: npts.c npts.h time_npts.c arraytools.c arraytools.h
	mpicc -g -o time_npts.run npts.c time_npts.c arraytools.c -I . -lm

clean:
	rm -f *.o time_npts.run *.png *.txt
