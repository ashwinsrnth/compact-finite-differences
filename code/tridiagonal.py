import pyopencl as cl
import numpy as np
import time

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

print device

kernel_text = """
__kernel void compactTDMA(__global double *a_d,
                                __global double *b_d,
                                __global double *c_d,
                                __global double *d_d,
                                __global double *x_d,
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

    x_d[block_end] = d_d[block_end];

    for (int i=block_size-2; i >= 0; i--)
    {
        x_d[block_start+i] = d_d[block_start+i] - c2_d[i]*x_d[block_start+i+1];
    }
}

"""

if 'NVIDIA' in platform.name:
    kernel_text = '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n' + kernel_text
    prg = cl.Program(ctx, kernel_text).build(options=['-cl-nv-arch sm_35'])
   
else:
    prg = cl.Program(ctx, kernel_text).build(options=['-O2'])

def solve_many_small_systems(a, b, c, d, num_systems, system_size):


    t1 = time.time()
    dfdx = np.zeros(num_systems*system_size, dtype=np.float64)
    t2 = time.time()

    print 'Initial allocation: ', t2-t1

    a_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, system_size*8)
    b_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, system_size*8)
    c_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, system_size*8)
    c2_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, system_size*8)
    d_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)
    dfdx_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, num_systems*system_size*8)

    t1 = time.time()

    evt1 = cl.enqueue_copy(queue, a_g, a)
    evt2 = cl.enqueue_copy(queue, b_g, b)
    evt3 = cl.enqueue_copy(queue, c_g, c)
    evt4 = cl.enqueue_copy(queue, d_g, d)
    evt5 = cl.enqueue_copy(queue, c2_g, c)
    evt6 = cl.enqueue_copy(queue, dfdx_g, dfdx)

    evt1.wait()
    evt2.wait()
    evt3.wait()
    evt4.wait()
    evt5.wait()
    evt6.wait()

    t2 = time.time()

    print 'Time for buffer copies: ', t2-t1

    t1 = time.time()

    evt = prg.compactTDMA(queue, [num_systems], None,
        a_g, b_g, c_g, d_g, dfdx_g, c2_g,
            np.int32(system_size))

    evt.wait()

    t2 = time.time()

    print 'Time for solve: ', t2-t1

    evt = cl.enqueue_copy(queue, dfdx, dfdx_g)
    evt.wait()

    return dfdx
