clean:
	rm -f *.o run

run: main.o
	nvcc -o run -arch=sm_35 main.o -lcublas -lcusparse

main.o:	main.cpp
	nvcc -c -O2 main.cpp -arch=sm_35
