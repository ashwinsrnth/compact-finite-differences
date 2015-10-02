MPIFLAGS=--mca btl_openib_warn_nonexistent_if 0 --mca btl_openib_want_cuda_gdr 0 --mca pml ob1

test:
	make test_1d
	make test_3d
	rm -f *.pyc

test_1d:
	@echo "testing 1d case with 3 procs: "
	mpiexec -n 3 ${MPIFLAGS} nosetests test_1d.py

test_3d:
	@echo "testing 3d case with 27 procs: "
	mpiexec -n 27 ${MPIFLAGS} nosetests test_3d.py

clean:
	rm -f *.pyc
