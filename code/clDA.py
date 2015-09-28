from mpi4py import MPI
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import time

class GpuDA:

    def __init__(self, comm, local_dims, proc_sizes, stencil_width):        
        self.comm = comm
        self.local_dims = local_dims
        self.proc_sizes = proc_sizes
        self.stencil_width = stencil_width
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        assert(isinstance(comm, MPI.Cartcomm))
        assert(self.size == reduce(lambda a,b: a*b, proc_sizes))
        self._create_halo_arrays()
   
    def global_to_local(self, global_array, local_array):
        # Update the local array (which includes ghost points)
        # from the global array (which does not)

        npz, npy, npx = self.proc_sizes
        nz, ny, nx = self.local_dims
        zloc, yloc, xloc = self.comm.Get_topo()[2]
        sw = self.stencil_width

        # copy inner elements:
        self._copy_global_to_local(global_array, local_array)

        # copy from arrays to send halos:
        self._copy_array_to_halo(global_array, self.left_send_halo, [nz, ny, sw], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.right_send_halo, [nz, ny, sw], [0, 0, nx-1])

        self._copy_array_to_halo(global_array, self.bottom_send_halo, [nz, sw, nx], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.top_send_halo, [nz, sw, nx], [0, ny-1, 0])

        self._copy_array_to_halo(global_array, self.front_send_halo, [sw, ny, nx], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.back_send_halo, [sw, ny, nx], [nz-1, 0, 0])

        # perform swaps in x-direction
        sendbuf = [self.right_send_halo.gpudata.as_buffer(self.right_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.left_recv_halo.gpudata.as_buffer(self.left_recv_halo.nbytes), MPI.DOUBLE]
        req1 = self._forward_swap(sendbuf, recvbuf, self.rank-1, self.rank+1, xloc, npx, 10)

        sendbuf = [self.left_send_halo.gpudata.as_buffer(self.left_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.right_recv_halo.gpudata.as_buffer(self.right_recv_halo.nbytes), MPI.DOUBLE]
        req2 = self._backward_swap(sendbuf, recvbuf, self.rank+1, self.rank-1, xloc, npx, 20)

        # perform swaps in y-direction:
        sendbuf = [self.top_send_halo.gpudata.as_buffer(self.top_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.bottom_recv_halo.gpudata.as_buffer(self.bottom_recv_halo.nbytes), MPI.DOUBLE]
        req3 = self._forward_swap(sendbuf, recvbuf, self.rank-npx, self.rank+npx, yloc, npy, 30)
       
        sendbuf = [self.bottom_send_halo.gpudata.as_buffer(self.bottom_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.top_recv_halo.gpudata.as_buffer(self.top_recv_halo.nbytes), MPI.DOUBLE]
        req4 = self._backward_swap(sendbuf, recvbuf, self.rank+npx, self.rank-npx, yloc, npy, 40)

        # perform swaps in z-direction:
        sendbuf = [self.back_send_halo.gpudata.as_buffer(self.back_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.front_recv_halo.gpudata.as_buffer(self.front_recv_halo.nbytes), MPI.DOUBLE]
        req5 = self._forward_swap(sendbuf, recvbuf, self.rank-npx*npy, self.rank+npx*npy, zloc, npz, 50)
       
        sendbuf = [self.front_send_halo.gpudata.as_buffer(self.front_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.back_recv_halo.gpudata.as_buffer(self.back_recv_halo.nbytes), MPI.DOUBLE]
        req6 = self._backward_swap(sendbuf, recvbuf, self.rank+npx*npy, self.rank-npx*npy, zloc, npz, 60)

        requests = [req for req in  [req1, req2, req3, req4, req5, req6] if req != None]
        MPI.Request.Waitall(requests, [MPI.Status()]*len(requests))

        # copy from recv halos to local_array:
        if self.has_neighbour('left'):
            self._copy_halo_to_array(self.left_recv_halo, local_array, [nz, ny, sw], [sw, sw, 0])
        
        if self.has_neighbour('right'):
            self._copy_halo_to_array(self.right_recv_halo, local_array, [nz, ny, sw], [sw, sw, 2*sw+nx-1])

        if self.has_neighbour('bottom'):
            self._copy_halo_to_array(self.bottom_recv_halo, local_array, [nz, sw, nx], [sw, 0, sw])
        
        if self.has_neighbour('top'):
            self._copy_halo_to_array(self.top_recv_halo, local_array, [nz, sw, nx], [sw, 2*sw+ny-1, sw])

        if self.has_neighbour('front'):
            self._copy_halo_to_array(self.front_recv_halo, local_array, [sw, ny, nx], [0, sw, sw])
        
        if self.has_neighbour('back'):
            self._copy_halo_to_array(self.back_recv_halo, local_array, [sw, ny, nx], [2*sw+nz-1, sw, sw])
        
    def local_to_global(self, local_array, global_array):

        # Update a global array (no ghost values)
        # from a local array (which contains ghost values).
        # This does *not* involve any communication.

        self._copy_local_to_global(local_array, global_array)


    def _forward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs, tag):
        
        # Perform swap in the +x, +y or +z direction
        
        if loc > 0 and loc < dimprocs-1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)

        elif loc == 0 and dimprocs > 1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = None

        elif loc == dimprocs-1 and dimprocs > 1:
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)
            
        return req

    def _backward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs, tag):
        
        # Perform swap in the -x, -y or -z direction
        
        if loc > 0 and loc < dimprocs-1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)
        
        elif loc == 0 and dimprocs > 1:
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)

        elif loc == dimprocs-1 and dimprocs > 1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = None

        return req

    def _create_halo_arrays(self):

        # Allocate space for the halos: two per face,
        # one for sending and one for receiving.

        nz, ny, nx = self.local_dims
        sw = self.stencil_width
        # create two halo regions for each face, one holding
        # the halo values to send, and the other holding
        # the halo values to receive.

        self.left_recv_halo = gpuarray.empty([nz,ny,sw], dtype=np.float64)
        self.left_send_halo = self.left_recv_halo.copy()
        self.right_recv_halo = self.left_recv_halo.copy()
        self.right_send_halo = self.left_recv_halo.copy()
    
        self.bottom_recv_halo = gpuarray.empty([nz,sw,nx], dtype=np.float64)
        self.bottom_send_halo = self.bottom_recv_halo.copy()
        self.top_recv_halo = self.bottom_recv_halo.copy()
        self.top_send_halo = self.bottom_recv_halo.copy()

        self.back_recv_halo = gpuarray.empty([sw,ny,nx], dtype=np.float64)
        self.back_send_halo = self.back_recv_halo.copy()
        self.front_recv_halo = self.back_recv_halo.copy()
        self.front_send_halo = self.back_recv_halo.copy()

    def _copy_array_to_halo(self, array, halo, copy_dims, copy_offsets, dtype=np.float64):

        # copy from 3-d array to 2-d halo
        #
        # Paramters:
        # array, halo:  gpuarrays involved in the copy.
        # copy_dims: number of elements to copy in (z, y, x) directions
        # copy_offsets: offsets at the source in (z, y, x) directions
        
        nz, ny, nx = self.local_dims 
        d, h, w  = copy_dims
        z_offs, y_offs, x_offs = copy_offsets
        
        typesize = array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(array.gpudata)
        copier.set_dst_device(halo.gpudata)

        copier.src_x_in_bytes = x_offs*typesize
        copier.src_y = y_offs
        copier.src_z = z_offs

        copier.src_pitch = array.strides[1]
        copier.dst_pitch = halo.strides[1]
        copier.src_height = ny
        copier.dst_height = h


        copier.width_in_bytes = w*typesize
        copier.height = h
        copier.depth = d

        # perform the copy:
        copier()

    def _copy_halo_to_array(self, halo, array, copy_dims, copy_offsets, dtype=np.float64):
        
        # copy from 2-d halo to 3-d array
        #
        # Parameters:
        # halo, array:  gpuarrays involved in the copy
        # copy_dims: number of elements to copy in (z, y, x) directions
        # copy_offsets: offsets at the destination in (z, y, x) directions
        
        nz, ny, nx = self.local_dims
        sw = self.stencil_width
        d, h, w = copy_dims
        z_offs, y_offs, x_offs = copy_offsets

        typesize = array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(halo.gpudata)
        copier.set_dst_device(array.gpudata)

        # this time, offsets are at the destination:
        copier.dst_x_in_bytes = x_offs*typesize
        copier.dst_y = y_offs
        copier.dst_z = z_offs

        copier.src_pitch = halo.strides[1]
        copier.dst_pitch = array.strides[1]
        copier.src_height = h
        copier.dst_height = ny+2*sw

        copier.width_in_bytes = w*typesize
        copier.height = h
        copier.depth = d
        
        # perform the copy:
        copier()

    def _copy_global_to_local(self, global_array, local_array, dtype=np.float64):

        nz, ny, nx = self.local_dims
        sw = self.stencil_width
      
        typesize = global_array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(global_array.gpudata)
        copier.set_dst_device(local_array.gpudata)

        # offsets 
        copier.dst_x_in_bytes = sw*typesize
        copier.dst_y = sw
        copier.dst_z = sw

        copier.src_pitch = global_array.strides[1] 
        copier.dst_pitch = local_array.strides[1]
        copier.src_height = ny
        copier.dst_height = ny+2*sw

        copier.width_in_bytes = nx*typesize
        copier.height = ny
        copier.depth = nz

        copier()

    def _copy_local_to_global(self, local_array, global_array, dtype=np.float64):

        nz, ny, nx = self.local_dims
        sw = self.stencil_width

        typesize = global_array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(local_array.gpudata)
        copier.set_dst_device(global_array.gpudata)

        # offsets
        copier.src_x_in_bytes = sw*typesize
        copier.src_y = sw
        copier.src_z = sw

        copier.src_pitch = local_array.strides[1]
        copier.dst_pitch = global_array.strides[1]
        copier.src_height = ny+2*sw
        copier.dst_height = ny

        copier.width_in_bytes = nx*typesize
        copier.height = ny
        copier.depth = nz

        copier()


    def has_neighbour(self, side):
        
        # Check that the processor has a
        # neighbour on a specified side
        # side can be 'left', 'right', 'top' or 'bottom'
        
        npz, npy, npx = self.comm.Get_topo()[0]
        mz, my, mx = self.comm.Get_topo()[2]
        
        if side == 'left' and mx > 0:
            return True
        
        elif side == 'right' and mx < npx-1:
            return True

        elif side == 'bottom' and my > 0:
            return True
        
        elif side == 'top' and my < npy-1:
            return True

        elif side == 'front' and mz > 0:
            return True

        elif side == 'back' and mz < npz-1:
            return True

        else:
            return False
       
