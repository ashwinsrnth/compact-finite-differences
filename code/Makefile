
phony:
	@echo
	@echo Make can be run with the following targets
	@echo test: Run all tests
	@echo clean: Clean up the directory 
	@echo
test:
	mpiexec -n 8 python test_compact.py

clean:
	rm -f *.png *.pyc
