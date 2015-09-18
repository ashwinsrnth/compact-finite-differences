time_npts: npts.c npts.h time_npts.c arraytools.c arraytools.h
	mpicc -O2 -o time_npts.run npts.c time_npts.c arraytools.c -I . -lm
test_npts: npts.c npts.h test_npts.c arraytools.c arraytools.h
	mpicc -O2 -o test_npts.run npts.c test_npts.c arraytools.c -I . -lm

clean:
	rm -f *.o test_npts.run *.png *.txt
