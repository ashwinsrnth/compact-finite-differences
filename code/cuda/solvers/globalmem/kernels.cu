extern "C"{
__global__ void globalForwardReduction(const double *a_d,
                                const double *b_d,
                                const double *c_d,
                                double *d_d,
                                const double *k1_d,
                                const double *k2_d,
                                const double *b_first_d,
                                const double *k1_first_d,
                                const double *k1_last_d,
                                const int nx,
                                const int ny,
                                const int nz,
                                int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
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

        m = log2((float)stride) - 1;
        n = log2((float)stride); // the last element

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

        idx = log2((float)stride) - 1;
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

__global__ void globalBackSubstitution(const double *a_d,
                                    const double *b_d,
                                    const double *c_d,
                                    double *d_d,
                                    const double *b_first_d,
                                    const double b1,
                                    const double c1,
                                    const double ai,
                                    const double bi,
                                    const double ci,
                                    const int nx,
                                    const int ny,
                                    const int nz,
                                    const int stride)
{
    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int giz = blockIdx.z*blockDim.z + threadIdx.z;
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
        // rint rounds to the nearest integer
        idx = rint(log2((double)stride)) - 2;
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
}
