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

a = np.zeros([128, 128, 128], dtype=np.float64)
b = np.random.rand(128, 128, 128)
f = np.zeros([128, 128, 2], dtype=np.float64)

a_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, a.size*8)
f_g = cl.Buffer(context, cl.mem_flags.READ_WRITE, f.size*8)

evt = cl.enqueue_copy(queue, a_g, a)
evt.wait()

# time a memcpy:
t1 = time.time()
evt = cl.enqueue_copy(queue, a_g, b)
evt.wait()
t2 = time.time()
print t2-t1

# time copying the faces
t1 = time.time()
prg.copyFaces(queue,
        [1, 128, 128], None, 
            a_g, f_g, np.int32(128), np.int32(128), np.int32(128))
evt = cl.enqueue_copy(queue, f, f_g)
evt.wait()
t2 = time.time()

print t2-t1
