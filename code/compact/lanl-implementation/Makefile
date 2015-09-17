
test_npts: npts.c npts.h test_npts.c arraytools.c arraytools.h
	mpicc -g -o test_npts.run npts.c test_npts.c arraytools.c -I . -lm

clean:
	rm -f *.o test_npts.run *.png *.txt
