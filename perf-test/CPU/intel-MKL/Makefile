
clean:
	rm run *.o

run: main.o
	icc -o run main.o -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -O2

main.o:	main.cpp
	icc -c main.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -O2
