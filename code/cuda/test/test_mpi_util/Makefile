MPIFLAGS=--mca btl_openib_warn_nonexistent_if 0 --mca btl_openib_want_cuda_gdr 0 --mca pml ob1

test:
	make test_1d
	make test_3d
	make test_dof
	make test_dof_3d
	make test_misc
	rm -f *.pyc

test_1d:
	@echo "testing 1d case with 3 procs: "
	mpiexec -n 3 ${MPIFLAGS} nosetests test_1d.py

test_3d:
	@echo "testing 3d case with 27 procs: "
	mpiexec -n 27 ${MPIFLAGS} nosetests test_3d.py

test_dof:
	@echo "testing 1d case with 3 procs and stencil width 2: "
	mpiexec -n 3 ${MPIFLAGS} nosetests test_sw.py

test_dof_3d:
	@echo "testing 3d case with 27 procs and stencil width 2: "
	mpiexec -n 27 ${MPIFLAGS} nosetests test_sw_3d.py

test_misc:
	@echo "testing miscellaneous functions in mpi_util: "
	mpiexec -n 8 ${MPIFLAGS} python test_misc.py

clean:
	rm -f *.pyc
