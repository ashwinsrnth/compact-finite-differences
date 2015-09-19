

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

__kernel void sumSolutionsdfdx2D(__global double* x_R_d,
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

    int iy = get_global_id(0);
    int iz = get_global_id(1);
    int i3d, i2d;
    double a, b;

    i2d = iz*ny + iy;
    a = alpha[i2d];
    b = beta[i2d];

    for (int ix=0; ix<nx; ix++) {
        i3d = iz*(nx*ny) + iy*nx + ix;
        x_R_d[i3d] = x_R_d[i3d] + a*x_UH_d[ix] + b*x_LH_d[ix];
    }
}


__kernel void NCyclicReduction(__global double *a_g,
                               __global double *b_g,
                               __global double *c_g,
                               __global double *d_g,
                               int nx,
                               int ny,
                               int nz,
                               int block_size,
                               __local double *a_l,
                               __local double *b_l,
                               __local double *c_l,
                               __local double *d_l) {

    /*
        Solve several systems by cyclic reduction,
        each of size block_size.
    */
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    int lid = get_local_id(0);
    int i, m, n;
    int stride;
    int i3d = iz*(nx*ny) + iy*nx + ix;

    double k1, k2;
    double d_m, d_n;

    /* each block reads its portion to shared memory */
    a_l[lid] = a_g[ix];
    b_l[lid] = b_g[ix];
    c_l[lid] = c_g[ix];
    d_l[lid] = d_g[i3d];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    /* solve the block in shared memory */
    stride = 1;
    for (int step=0; step<native_log2((float) nx); step++) {
        stride = stride*2;

        if (lid < nx/stride) {
            i = (stride-1) + lid*(stride);

            if (stride == nx) {
                m = nx/2 - 1;
                n = nx - 1;

                d_m = (d_l[m]*b_l[n] - c_l[m]*d_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_n = (b_l[m]*d_l[n] - d_l[m]*a_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_l[m] = d_m;
                d_l[n] = d_n;
            }

            else {
                if (i == (nx-1)) {
                    k1 = a_l[i]/b_l[i-stride/2];
                    a_l[i] = -a_l[i-stride/2]*k1;
                    b_l[i] = b_l[i] - c_l[i-stride/2]*k1;
                    d_l[i] = d_l[i] - d_l[i-stride/2]*k1;
                }
                else {
                    k1 = a_l[i]/b_l[i-stride/2];
                    k2 = c_l[i]/b_l[i+stride/2];
                    a_l[i] = -a_l[i-stride/2]*k1;
                    b_l[i] = b_l[i] - c_l[i-stride/2]*k1 - a_l[i+stride/2]*k2;
                    c_l[i] = -c_l[i+stride/2]*k2;
                    d_l[i] = d_l[i] - d_l[i-stride/2]*k1 - d_l[i+stride/2]*k2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    
    for (int step=0; step<native_log2((float) nx)-1; step++) {
        stride = stride/2;

        if (lid < nx/stride){
            i = (stride/2-1) + lid*stride;

            if (i < stride) {
                d_l[i] = (d_l[i] - c_l[i]*d_l[i+stride/2])/b_l[i];
            }

            else {
                d_l[i] = (d_l[i] - a_l[i]*d_l[i-stride/2] - c_l[i]*d_l[i+stride/2])/b_l[i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* write from shared memory to x_d */
    d_g[i3d] = d_l[lid];
    barrier(CLK_GLOBAL_MEM_FENCE);
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

__kernel void MultiNCyclicReduction(__global double *a_g,
                               __global double *b_g,
                               __global double *c_g,
                               __global double *d_g,
                               int nx,
                               int ny,
                               int nz,
                               int bx,
                               int by,
                               __local double *a_l,
                               __local double *b_l,
                               __local double *c_l,
                               __local double *d_l) {

    /*
        Solve several systems by cyclic reduction,
        each of size block_size.
    */
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    int lix = get_local_id(0);
    int liy = get_local_id(1);
    int liz = get_local_id(2);
    int i, m, n;
    int stride;

    int i3d = iz*(nx*ny) + iy*nx + ix;
    int li3d = liz*(bx*by) + liy*bx + lix;
    int lix0 = liz*(bx*by) + liy*bx + 0;

    double k1, k2;
    double d_m, d_n;

    /* each block reads its portion to shared memory */
    a_l[li3d] = a_g[ix];
    b_l[li3d] = b_g[ix];
    c_l[li3d] = c_g[ix];
    d_l[li3d] = d_g[i3d];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    /* solve the block in shared memory */
    stride = 1;
    for (int step=0; step<native_log2((float) nx); step++) {
        stride = stride*2;

        if (lix < nx/stride) {
            
            i = (stride-1) + lix*stride;
            ix = lix0 + i;

            if (stride == nx) {
                m = lix0 + nx/2 - 1;
                n = lix0 + nx - 1;

                d_m = (d_l[m]*b_l[n] - c_l[m]*d_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_n = (b_l[m]*d_l[n] - d_l[m]*a_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_l[m] = d_m;
                d_l[n] = d_n;
            }

            else {
                if (i == (nx-1)) {
                    ix = lix0 + i;
                    k1 = a_l[ix]/b_l[ix-stride/2];
                    a_l[ix] = -a_l[ix-stride/2]*k1;
                    b_l[ix] = b_l[ix] - c_l[ix-stride/2]*k1;
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1;
                }
                else {
                    k1 = a_l[ix]/b_l[ix-stride/2];
                    k2 = c_l[ix]/b_l[ix+stride/2];
                    a_l[ix] = -a_l[ix-stride/2]*k1;
                    b_l[ix] = b_l[ix] - c_l[ix-stride/2]*k1 - a_l[ix+stride/2]*k2;
                    c_l[ix] = -c_l[ix+stride/2]*k2;
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1 - d_l[ix+stride/2]*k2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    
    for (int step=0; step<native_log2((float) nx)-1; step++) {
        stride = stride/2;

        if (lix < nx/stride){
            i = (stride/2-1) + lix*stride;
            ix = lix0 + i;

            if (i < stride) {
                d_l[ix] = (d_l[ix] - c_l[ix]*d_l[ix+stride/2])/b_l[ix];
            }

            else {
                d_l[ix] = (d_l[ix] - a_l[ix]*d_l[ix-stride/2] - c_l[ix]*d_l[ix+stride/2])/b_l[ix];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* write from shared memory to x_d */
    d_g[i3d] = d_l[li3d];
    barrier(CLK_GLOBAL_MEM_FENCE);
}


