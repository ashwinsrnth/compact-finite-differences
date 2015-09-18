import pyopencl as cl
import kernels
import numpy as np
import time

# compare the performance of the copyFaces kernel
# v/s an actual memcpy


platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
prg = kernels.get_kernels(context)

N = 512

a = np.zeros([N, N, N], dtype=np.float64)
b = np.random.rand(N, N, N)
f = np.zeros([N, N, 2], dtype=np.float64)

a_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, a.size*8)
f_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, f.size*8)

evt = cl.enqueue_copy(queue, a_g, a)
evt.wait()

# time a memcpy:
for i in range(5):
    t1 = time.time()
    evt = cl.enqueue_copy(queue, a_g, b)
    evt.wait()
    t2 = time.time()
    evt = prg.copyFaces(queue,
            [1, N, N], None, 
                a_g, f_g, np.int32(N), np.int32(N), np.int32(N))
    evt.wait()
    t3 = time.time()
    evt = cl.enqueue_copy(queue, f, f_g)
    evt.wait()
    t4 = time.time()
    print 'Full buffer copy: ', t2-t1
    print 'Kernel: ', t3-t2
    print 'Copy: ', t4-t3
    print 'Total: ', t4-t2
