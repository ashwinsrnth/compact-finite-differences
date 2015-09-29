__kernel void computeRHS(
                        __global double *f_local_d,
                        __global double *rhs_x_d,
                        __global double *rhs_y_d,
                        __global double *rhs_z_d,
                        double dx,
                        double dy,
                        double dz,
                        int mx, 
                        int npx,
                        int my,
                        int npy,
                        int mz,
                        int npz)
{
    /*  
    Computes the RHS for solving for the derivatives
    of a function f. f_local is the "local" part of
    the function which includes ghost points.

    dx & dy are the spacing.

    nx, ny, nz define the size of d. f_local is shaped
    [nz+2, ny+2, nx+2]

    mi nd npi together decide if we are evaluating
    at a boundary.
    */  

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    int nz = get_global_size(2);

    int i = iz*(nx*ny) + iy*nx + ix; 
    int iloc = (iz+1)*((nx+2)*(ny+2)) + (iy+1)*(nx+2) + (ix+1);

    rhs_x_d[i] = (3./(4*dx))*(f_local_d[iloc+1] - f_local_d[iloc-1]);
    rhs_y_d[i] = (3./(4*dy))*(f_local_d[iloc+(nx+2)] - f_local_d[iloc-(nx+2)]);
    rhs_z_d[i] = (3./(4*dz))*(f_local_d[iloc+(nx+2)*(ny+2)] - f_local_d[iloc-(nx+2)*(ny+2)]);

    if (mx == 0) {
        if (ix == 0) {
            rhs_x_d[i] = (1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+1] + f_local_d[iloc+2]);
        }
    }

    if (mx == npx-1) {
        if (ix == nx-1) {
            rhs_x_d[i] = -(1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-1] + f_local_d[iloc-2]);
        }
    }

    if (my == 0) {
        if (iy == 0) {
            rhs_y_d[i] = (1./(2*dy))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+(nx+2)] + f_local_d[iloc+2*(nx+2)]);
        }
    }

    if (my == npy-1) {
        if (iy == ny-1) {
            rhs_y_d[i] = -(1./(2*dy))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-(nx+2)] + f_local_d[iloc-2*(nx+2)]);
        }
    }


    if (mz == 0) {
        if (iz == 0) {
            rhs_z_d[i] = (1./(2*dz))*(-5*f_local_d[iloc] + \
                4*f_local_d[iloc+(nx+2)*(ny+2)] + f_local_d[iloc+2*(nx+2)*(ny+2)]);
        }
    }

    if (mz == npz-1) {
        if (iz == nz-1) {
            rhs_z_d[i] = -(1./(2*dz))*(-5*f_local_d[iloc] + \
                4*f_local_d[iloc-(nx+2)*(ny+2)] + f_local_d[iloc-2*(nx+2)*(ny+2)]);
        }
    }

}

__kernel void sumSolutions(__global double* x_R_d,
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

    x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
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
__kernel void pThomasKernel(__global double *a_d,
                                __global double *b_d,
                                __global double *c_d,
                                __global double *c2_d,
                                __global double *d_d,
                                int block_size)
{
    /*
    Solves many small tridiagonal systems
    using a pThomas (thread-parallel Thomas algorithm)
    */

    int gid = get_global_id(0);
    int block_start = gid*block_size;
    double bmac;

    /* do a serial TDMA on the local system */

    c2_d[0] = c_d[0]/b_d[0]; // we need c2_d, because every thread will overwrite c_d[0] otherwise
    d_d[block_start] = d_d[block_start]/b_d[0];

    for (int i=1; i<block_size; i++)
    {
        bmac = b_d[i] - a_d[i]*c2_d[i-1];
        c2_d[i] = c_d[i]/bmac;
        d_d[block_start+i] = (d_d[block_start+i] - a_d[i]*d_d[block_start+i-1])/bmac;
    }

    for (int i=block_size-2; i >= 0; i--)
    {
        d_d[block_start+i] = d_d[block_start+i] - c2_d[i]*d_d[block_start+i+1];
    }
}

   
__kernel void singleLineCyclicReduction(__global double *a_g,
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

__kernel void multiLineCyclicReduction(__global double *a_g,
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
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int lix = get_local_id(0);
    int liy = get_local_id(1);
    int liz = get_local_id(2);
    int i, m, n, ix;
    int stride;

    int i3d = giz*(nx*ny) + giy*nx + gix;
    int li3d = liz*(bx*by) + liy*bx + lix;
    int lix0 = liz*(bx*by) + liy*bx + 0;

    double k1, k2;
    double d_m, d_n;

    /* each block reads its portion to shared memory */
    a_l[lix] = a_g[gix];
    b_l[lix] = b_g[gix];
    c_l[lix] = c_g[gix];
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
                m = nx/2 - 1;
                n = nx - 1;

                d_m = (d_l[lix0+m]*b_l[n] - c_l[m]*d_l[lix0+n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_n = (b_l[m]*d_l[lix0+n] - d_l[lix0+m]*a_l[n])/(b_l[m]*b_l[n] - c_l[m]*a_l[n]);
                d_l[lix0+m] = d_m;
                d_l[lix0+n] = d_n;
            }

            else {
                if (i == (nx-1)) {
                    k1 = a_l[i]/b_l[i-stride/2];
                    a_l[i] = -a_l[i-stride/2]*k1;
                    b_l[i] = b_l[i] - c_l[i-stride/2]*k1;
                    d_l[ix] = d_l[ix] - d_l[ix-stride/2]*k1;
                }
                else {
                    k1 = a_l[i]/b_l[i-stride/2];
                    k2 = c_l[i]/b_l[i+stride/2];
                    a_l[i] = -a_l[i-stride/2]*k1;
                    b_l[i] = b_l[i] - c_l[i-stride/2]*k1 - a_l[i+stride/2]*k2;
                    c_l[i] = -c_l[i+stride/2]*k2;
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
                d_l[ix] = (d_l[ix] - c_l[i]*d_l[ix+stride/2])/b_l[i];
            }

            else {
                d_l[ix] = (d_l[ix] - a_l[i]*d_l[ix-stride/2] - c_l[i]*d_l[ix+stride/2])/b_l[i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* write from shared memory to x_d */
    d_g[i3d] = d_l[li3d];
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void globalForwardReduction_x(__global double *a_d,
                               __global double *b_d,
                               __global double *c_d,
                               __global double *d_d,
                               __global double *k1_d,
                               __global double *k2_d,
                               __global double *b_first_d,
                               __global double *k1_first_d,
                               __global double *k1_last_d,
                               int nx,
                               int ny,
                               int nz,
                               int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int m, n;
    int idx;
    int gi3d, gi3d0;
    double x_m, x_n;

    gi3d = giz*(nx*ny) + giy*nx + gix;
    gi3d0 = giz*(nx*ny) + giy*nx + 0;

    // forward reduction
    if (stride == nx)
    {
        stride /= 2;

        // note that just log2() fails on GPUs for some reason
        m = native_log2((float)stride) - 1;
        n = native_log2((float)stride); // the last element

        x_m = (d_d[gi3d0 + stride-1]*b_d[n] - c_d[m]*d_d[gi3d0 + 2*stride-1])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        x_n = (b_first_d[m]*d_d[gi3d0 + 2*stride-1] - d_d[gi3d0 + stride-1]*a_d[n])/ \
                        (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        d_d[gi3d0 + stride-1] = x_m;
        d_d[gi3d0 + 2*stride-1] = x_n;
    }
    else
    {
        i = (stride-1) + gix*stride;
        gi3d = gi3d0 + i;

        idx = native_log2((float)stride) - 1;
        if (gix == 0)
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_first_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
        else if (i == (nx-1))
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_last_d[idx];
        }
        else
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
    }
}


__kernel void globalBackSubstitution_x(__global double *a_d,
                                   __global double *b_d,
                                   __global double *c_d,
                                   __global double *d_d,
                                   __global double *b_first_d,
                                   double b1,
                                   double c1,
                                   double ai,
                                   double bi,
                                   double ci,
                                   int nx,
                                   int ny,
                                   int nz,
                                   int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int idx;
    int gi3d, gi3d0;

    gi3d0 = giz*(nx*ny) + giy*nx + 0;

    i = (stride/2-1) + gix*stride;
    gi3d = gi3d0 + i;

    if (stride == 2)
    {
        if (i == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c1*d_d[gi3d + 1])/b1;
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - (ai)*d_d[gi3d - 1] - (ci)*d_d[gi3d + 1])/bi;
        }
    }
    else
    {
        // note that just log2() fails on GPUs for some reason
        idx = native_log2((float)stride) - 2;
        if (gix == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c_d[idx]*d_d[gi3d + stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - a_d[idx]*d_d[gi3d - stride/2] - c_d[idx]*d_d[gi3d + stride/2])/b_d[idx];
        }
    }
}


__kernel void globalForwardReduction_y(__global double *a_d,
                               __global double *b_d,
                               __global double *c_d,
                               __global double *d_d,
                               __global double *k1_d,
                               __global double *k2_d,
                               __global double *b_first_d,
                               __global double *k1_first_d,
                               __global double *k1_last_d,
                               int nx,
                               int ny,
                               int nz,
                               int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int m, n;
    int idx;
    int gi3d, gi3d0;
    double x_m, x_n;

    gi3d = giz*(nx*ny) + giy*ny + gix;
    gi3d0 = giz*(nx*ny) + 0 + gix;
    
    // forward reduction
    if (stride == nx*ny)
    {
       stride /= 2;

       // note that just log2() fails on GPUs for some reason
       m = native_log2((float)stride/nx) - 1;
       n = native_log2((float)stride/nx); // the last element

       x_m = (d_d[gi3d0 + stride-nx]*b_d[n] - c_d[m]*d_d[gi3d0 + 2*stride-nx])/ \
                       (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

       x_n = (b_first_d[m]*d_d[gi3d0 + 2*stride-nx] - d_d[gi3d0 + stride-nx]*a_d[n])/ \
                         (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        d_d[gi3d0 + stride-nx] = x_m;
        d_d[gi3d0 + 2*stride-nx] = x_n;
    }
    else
    {
        i = (stride-nx) + giy*stride;
        gi3d = gi3d0 + i;
        idx = native_log2((float)stride/nx) - 1;
        
        if (giy == 0)
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_first_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
        else if (i == nx*(ny-1))
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_last_d[idx];
        }
        else
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
    }
}


__kernel void globalBackSubstitution_y(__global double *a_d,
                                   __global double *b_d,
                                   __global double *c_d,
                                   __global double *d_d,
                                   __global double *b_first_d,
                                   double b1,
                                   double c1,
                                   double ai,
                                   double bi,
                                   double ci,
                                   int nx,
                                   int ny,
                                   int nz,
                                   int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int idx;
    int gi3d, gi3d0;

    gi3d0 = giz*(nx*ny) + 0 + gix;
    i = (stride/2-nx) + giy*stride;
    gi3d = gi3d0 + i;


    if (stride == 2*nx)
    {
        if (i == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c1*d_d[gi3d + nx])/b1;
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - (ai)*d_d[gi3d - nx] - (ci)*d_d[gi3d + nx])/bi;
        }
    }
    else
    {
        // note that just log2() fails on GPUs for some reason
        idx = native_log2((float)stride/nx) - 2;
        if (giy == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c_d[idx]*d_d[gi3d + stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - a_d[idx]*d_d[gi3d - stride/2] - c_d[idx]*d_d[gi3d + stride/2])/b_d[idx];
        }
    }
}

__kernel void globalForwardReduction_z(__global double *a_d,
                               __global double *b_d,
                               __global double *c_d,
                               __global double *d_d,
                               __global double *k1_d,
                               __global double *k2_d,
                               __global double *b_first_d,
                               __global double *k1_first_d,
                               __global double *k1_last_d,
                               int nx,
                               int ny,
                               int nz,
                               int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int m, n;
    int idx;
    int gi3d, gi3d0;
    double x_m, x_n;

    gi3d = giz*(nx*ny) + giy*ny + gix;
    gi3d0 = 0 + giy*ny + gix;
    
    // forward reduction
    if (stride == nx*ny*nz)
    {
       stride /= 2;

       // note that just log2() fails on GPUs for some reason
       m = native_log2((float)stride/(nx*ny)) - 1;
       n = native_log2((float)stride/(nx*ny)); // the last element

       x_m = (d_d[gi3d0 + stride-nx*ny]*b_d[n] - c_d[m]*d_d[gi3d0 + 2*stride-nx*ny])/ \
                       (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

       x_n = (b_first_d[m]*d_d[gi3d0 + 2*stride-nx*ny] - d_d[gi3d0 + stride-nx*ny]*a_d[n])/ \
                         (b_first_d[m]*b_d[n] - c_d[m]*a_d[n]);

        d_d[gi3d0 + stride-nx*ny] = x_m;
        d_d[gi3d0 + 2*stride-nx*ny] = x_n;
    }
    else
    {
        i = (stride-nx*ny) + giz*stride;
        gi3d = gi3d0 + i;
        idx = native_log2((float)stride/(nx*ny)) - 1;
        
        if (giz == 0)
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_first_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
        else if (i == nx*ny*(nz-1))
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_last_d[idx];
        }
        else
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d - stride/2]*k1_d[idx] - d_d[gi3d + stride/2]*k2_d[idx];
        }
    }
}


__kernel void globalBackSubstitution_z(__global double *a_d,
                                   __global double *b_d,
                                   __global double *c_d,
                                   __global double *d_d,
                                   __global double *b_first_d,
                                   double b1,
                                   double c1,
                                   double ai,
                                   double bi,
                                   double ci,
                                   int nx,
                                   int ny,
                                   int nz,
                                   int stride)
{
    int gix = get_global_id(0);
    int giy = get_global_id(1);
    int giz = get_global_id(2);
    int i;
    int idx;
    int gi3d, gi3d0;

    gi3d0 = 0 + giy*nx + gix;
    i = (stride/2-nx*ny) + giz*stride;
    gi3d = gi3d0 + i;


    if (stride == 2*nx*ny)
    {
        if (i == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c1*d_d[gi3d + nx*ny])/b1;
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - (ai)*d_d[gi3d - nx*ny] - (ci)*d_d[gi3d + nx*ny])/bi;
        }
    }
    else
    {
        // note that just log2() fails on GPUs for some reason
        idx = native_log2((float)stride/(nx*ny)) - 2;
        if (giz == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c_d[idx]*d_d[gi3d + stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - a_d[idx]*d_d[gi3d - stride/2] - c_d[idx]*d_d[gi3d + stride/2])/b_d[idx];
        }
    }
}
