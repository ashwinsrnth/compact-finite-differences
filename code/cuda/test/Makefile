export MPIEXECFLAGS=--mca pml ob1
phony:
	@echo
	@echo Make can be run with the following targets
	@echo test: Run all tests
	@echo clean: Clean up the directory 
	@echo
test_compact:
	mpiexec ${MPIEXECFLAGS} -n 8 python test_compact.py
demo:
	mpiexec ${MPIEXECFLAGS} -n 8 python demo.py
clean:
	rm -f *.png *.pyc
