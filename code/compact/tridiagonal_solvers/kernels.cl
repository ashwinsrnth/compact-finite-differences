__kernel void pThomasKernel(__global double *a_d,
                                __global double *b_d,
                                __global double *c_d,
                                __global double *d_d,
                                __global double *c2_d,
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


__kernel void globalForwardReduction(__global double *a_d,
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
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d-stride/2]*k1_first_d[idx] - d_d[gi3d+stride/2]*k2_d[idx];
        }
        else if (i == (nx-1))
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d-stride/2]*k1_last_d[idx];
        }
        else 
        {
            d_d[gi3d] = d_d[gi3d] - d_d[gi3d-stride/2]*k1_d[idx] - d_d[gi3d+stride/2]*k2_d[idx];
        }
    }
}

__kernel void globalBackSubstitution(__global double *a_d,
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
            d_d[gi3d] = (d_d[gi3d] - c1*d_d[gi3d+1])/b1;
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - (ai)*d_d[gi3d-1] - (ci)*d_d[gi3d+1])/bi;
        }
    }
    else
    {
        // note that just log2() fails on GPUs for some reason
        idx = native_log2((float)stride) - 2;
        if (gix == 0)
        {
            d_d[gi3d] = (d_d[gi3d] - c_d[idx]*d_d[gi3d+stride/2])/b_first_d[idx];
        }
        else
        {
            d_d[gi3d] = (d_d[gi3d] - a_d[idx]*d_d[gi3d-stride/2] - c_d[idx]*d_d[gi3d+stride/2])/b_d[idx];
        }
    }
}
