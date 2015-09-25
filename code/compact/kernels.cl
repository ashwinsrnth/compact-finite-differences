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


__kernel void computeRHSdfdy(__global double *f_local_d,
                        __global double *rhs_d,
                        double dy,
                        int nx,
                        int ny,
                        int nz,
                        int my,
                        int npy)
{
    /*
    Computes the RHS for solving for the y-derivative
    of a function f. f_local is the "local" part of
    the function which includes ghost points.

    dy is the spacing.

    nx, ny, nz define the size of d. f_local is shaped
    [nz+2, ny+2, nx+2]

    my and npy together decide if we are evaluating
    at a boundary.
    */

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    int i = iz*(nx*ny) + iy*nx + ix;
    int iloc = (iz+1)*((nx+2)*(ny+2)) + (iy+1)*(nx+2) + (ix+1);

    rhs_d[i] = (3./(4*dy))*(f_local_d[iloc+(nx+2)] - f_local_d[iloc-(nx+2)]);

    if (my == 0) {
        if (iy == 0) {
            rhs_d[i] = (1./(2*dy))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+(nx+2)] + f_local_d[iloc+2*(nx+2)]);
        }
    }

    if (my == npy-1) {
        if (iy == ny-1) {
            rhs_d[i] = -(1./(2*dy))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-(nx+2)] + f_local_d[iloc-2*(nx+2)]);
        }
    }
}


__kernel void sumSolutionsdfdx3D(__global double* x_R_d,
                            __global double* x_UH_d,
                            __global double* x_LH_d,
                            __global double* alpha,
                            __global double* beta,
                            int nx,
                            int ny,
                            int nz)
{
    /*
    Computes the sum of the solution x_R, x_UH and x_LH,
    where x_R is [nz, ny, nx] and x_LH & x_UH are [nx] sized.
    Performs the following:

    np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
    */
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    int i3d, i2d;

    i2d = iz*ny + iy;
    i3d = iz*(ny*nx) + iy*nx + ix;

    x_R_d[i3d] = x_R_d[i3d] + alpha[i2d]*x_UH_d[ix] + beta[i2d]*x_LH_d[ix];
}

__kernel void copyFaces(__global double* x,
            __global double* x_faces,
            int nx,
            int ny,
            int nz) {
    
    /*
    Copy the left and right face from the logically [nz, ny, nx] array x
    to a logically [nz, ny, 2] array x_faces 
    */

    int iy = get_global_id(1);
    int iz = get_global_id(2);

    int i_source;
    int i_dest;
    
    i_source = iz*(nx*ny) + iy*nx + 0;
    i_dest = iz*(2*ny) + iy*2 + 0;
    
    x_faces[i_dest] = x[i_source];

    i_source = iz*(nx*ny) + iy*nx + nx-1;
    i_dest = iz*(2*ny) + iy*2 + 1;
    
    x_faces[i_dest] = x[i_source];

}
