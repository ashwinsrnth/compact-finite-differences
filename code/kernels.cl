

__kernel void compactTDMA(__global double *a_d,
                                __global double *b_d,
                                __global double *c_d,
                                __global double *d_d,
                                __global double *c2_d,
                                int block_size)
{
    /*
    Solves many small systems arising from
    compact finite difference formulation.
    */

    int gid = get_global_id(0);
    int block_start = gid*block_size;
    int block_end = block_start + block_size - 1;

    /* do a serial TDMA on the local system */

    c2_d[0] = c_d[0]/b_d[0]; // we need c2_d, because every thread will overwrite c_d[0] otherwise
    d_d[block_start] = d_d[block_start]/b_d[0];

    for (int i=1; i<block_size; i++)
    {
        c2_d[i] = c_d[i]/(b_d[i] - a_d[i]*c2_d[i-1]);
        d_d[block_start+i] = (d_d[block_start+i] - a_d[i]*d_d[block_start+i-1])/(b_d[i] - a_d[i]*c2_d[i-1]);
    }

    for (int i=block_size-2; i >= 0; i--)
    {
        d_d[block_start+i] = d_d[block_start+i] - c2_d[i]*d_d[block_start+i+1];
    }
}



__kernel void computeRHSdfdx(__global double *f_local_d,
                        __global double *rhs_d,
                        double dx,
                        int nx,
                        int ny,
                        int nz,
                        int mx,
                        int npx)
{
    /*
    Computes the RHS for solving for the x-derivative
    of a function f. f_local is the "local" part of
    the function which includes ghost points.

    dx is the spacing.

    nx, ny, nz define the size of d. f_local is shaped
    [nz+2, ny+2, nx+2]

    mx and npx together decide if we are evaluating
    at a boundary.
    */

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    int i = iz*(nx*ny) + iy*nx + ix;
    int iloc = (iz+1)*((nx+2)*(ny+2)) + (iy+1)*(nx+2) + (ix+1);

    rhs_d[i] = (3./(4*dx))*(f_local_d[iloc+1] - f_local_d[iloc-1]);



    if (mx == 0) {
        if (ix == 0) {
            rhs_d[i] = (1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+1] + f_local_d[iloc+2]);
        }
    }

    if (mx == npx-1) {
        if (ix == nx-1) {
            rhs_d[i] = -(1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-1] + f_local_d[iloc-2]);
        }
    }
}
