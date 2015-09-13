#include <stdio.h>
#include <stdlib.h>

double getRandomDouble()
{
    return ((double)rand()/(double)RAND_MAX);
}

void initRandom(double* ary, int size)
{
    int i;
    /* Initialize ary with random doubles */
    for (i=0; i<size; i++) {
        ary[i] = getRandomDouble();
    }
}

void initZeros(double *ary, int size)
{
    int i;
    /* Initialize ary with zeros */
    for (i=0; i<size; i++) {
        ary[i] = 0.0;
    }
}

void initConst(double *ary, double c, int size)
{
    int i;
    /* Initialize ary with const */
    for (i=0; i<size; i++) {
        ary[i] = c;
    }
}

void initIntConst(int *ary, int c, int size)
{
    int i;
    /* Initialize ary with const */
    for (i=0; i<size; i++) {
        ary[i] = c;
    }
}

void initRange(int *ary, int start, int stop, int stride)
{
    int x;
    int idx = 0;
    /* Initialize ary with a range */
    for (x=start; x<stop; x+=stride)
    {
        ary[idx] = x;
        idx += 1;
    }
}

void printArray(double* ary, int size)
{
    int i;
    for (i=0; i<size; i++) {
        printf("%f\n", ary[i]);
    }
    printf("\n");
}

void printIntArray(int* ary, int size)
{
    int i;
    for (i=0; i<size; i++) {
        printf("%d\n", ary[i]);
    }
    printf("\n");
}

void negative(double* ary, int size)
{
    int i;
    for (i=0; i<size; i++) {
        ary[i] *= -1;
    }
}
