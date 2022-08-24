# cython: profile=False
# cython: language_level=3
import logging
import tempfile
import os
import shutil

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal
from pygeoprocessing.geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from pygeoprocessing.routing.routing import (
    _generate_read_bounds,
    _is_raster_path_band_formatted
)

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.pair cimport pair
from libcpp.set cimport set as cset
from libcpp.list cimport list as clist
from libcpp.stack cimport stack
from libc.time cimport time_t
from libc.time cimport time as ctime
cimport libc.math as cmath
from libcpp.stack cimport stack

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int _is_close(double x, double y, double abs_delta, double rel_delta):
    return abs(x-y) <= (abs_delta+rel_delta*abs(y))

cdef float _LOGGING_PERIOD = 10.0
cdef double IMPROBABLE_FLOAT_NODATA = -1.23789789e29
cdef struct FlowPixelType:
    int xi
    int yi
    int last_flow_dir
    double value
cdef int *D8_XOFFSET = [1, 1, 0, -1, -1, -1, 0, 1]
cdef int *D8_YOFFSET = [0, -1, -1, -1, 0, +1, +1, +1]
cdef int* D8_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]


# cmath is supposed to have M_SQRT2, but tests have been failing recently
# due to a missing symbol.
cdef double SQRT2 = cmath.sqrt(2)
cdef double PI = 3.141592653589793238462643383279502884
# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8
# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# These offsets are for the neighbor rows and columns according to the
# ordering: 3 2 1
#           4 x 0
#           5 6 7
cdef int *ROW_OFFSETS = [0, -1, -1, -1,  0,  1, 1, 1]
cdef int *COL_OFFSETS = [1,  1,  0, -1, -1, -1, 0, 1]

cdef float* FLOW_LENGTH = [1, SQRT2, 1, SQRT2, 1, SQRT2, 1, SQRT2]



cdef int is_close(double x, double y):
    return abs(x-y) <= (1e-8+1e-05*abs(y))

# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it
cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.  Copied from src/pygeoprocessing/routing/routing.pyx,
# revision 891288683889237cfd3a3d0a1f09483c23489fca.
cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xmod
    cdef int block_ymod
    cdef int block_xbits
    cdef int block_ybits
    cdef long raster_x_size
    cdef long raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef bytes raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Args:
            raster_path (char*): path to raster that has block sizes that are
                powers of 2. If not, an exception is raised.
            band_id (int): which band in `raster_path` to index. Uses GDAL
                notation that starts at 1.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            print(err_msg)
            raise ValueError(err_msg)
        self.band_id = band_id

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    self.block_xsize, self.block_ysize, raster_path))
            print(err_msg)
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) // self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) // self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = <bytes> raster_path
        self.write_mode = write_mode
        self.closed = 0

    def __dealloc__(self):
        """Deallocate _ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the _ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in _ManagedRaster will
            have undefined behavior.
        """
        if self.closed:
            return
        self.closed = 1
        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize))
        cdef double *double_buffer
        cdef int block_xi
        cdef int block_yi
        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize
        cdef int win_ysize

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff
        cdef int yoff

        cdef clist[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef clist[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.OpenEx(
            self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)

        # if we get here, we're in write_mode
        cdef cset[int].iterator dirty_itr
        while it != end:
            double_buffer = deref(it).second
            block_index = deref(it).first

            # write to disk if block is dirty
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr != self.dirty_blocks.end():
                self.dirty_blocks.erase(dirty_itr)
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for
                # cached array
                xoff = block_xi << self.block_xbits
                yoff = block_yi << self.block_ybits

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # clip window sizes if necessary
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (
                        xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (
                        yoff+win_ysize - self.raster_y_size)

                for xi_copy in xrange(win_xsize):
                    for yi_copy in xrange(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, long xi, long yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline double get(self, long xi, long yi):
        """Return the value of the pixel at `xi,yi`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index // self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
        cdef clist[BlockBufferPair] removed_value_list

        # determine the block aligned xoffset for read as array

        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize = self.block_xsize
        cdef int win_ysize = self.block_ysize

        # load a new block
        if xoff+win_xsize > self.raster_x_size:
            win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
        if yoff+win_ysize > self.raster_y_size:
            win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

        raster = gdal.OpenEx(self.raster_path, gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(
            numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in xrange(win_xsize):
            for yi_copy in xrange(win_ysize):
                double_buffer[(yi_copy<<self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.OpenEx(
                self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in xrange(win_xsize):
                        for yi_copy in xrange(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None


def calculate_sediment_deposition(
        mfd_flow_direction_path, e_prime_path, f_path, sdr_path,
        target_sediment_deposition_path):
    """Calculate sediment deposition layer.

    This algorithm outputs both sediment deposition (r_i) and flux (f_i)::

        r_i  =      dr_i  * (sum over j ∈ J of f_j * p(i,j)) + E'_i

        f_i  = (1 - dr_i) * (sum over j ∈ J of f_j * p(i,j)) + E'_i


                (sum over k ∈ K of SDR_k * p(i,k)) - SDR_i
        dr_i = --------------------------------------------
                              (1 - SDR_i)

    where:

    - ``p(i,j)`` is the proportion of flow from pixel ``i`` into pixel ``j``
    - ``J`` is the set of pixels that are immediate upslope neighbors of
      pixel ``i``
    - ``K`` is the set of pixels that are immediate downslope neighbors of
      pixel ``i``
    - ``E'`` is ``USLE * (1 - SDR)``, the amount of sediment loss from pixel
      ``i`` that doesn't reach a stream (``e_prime_path``)
    - ``SDR`` is the sediment delivery ratio (``sdr_path``)

    ``f_i`` is recursively defined in terms of ``i``'s upslope neighbors.
    The algorithm begins from seed pixels that are local high points and so
    have no upslope neighbors. It works downslope from each seed pixel,
    only adding a pixel to the stack when all its upslope neighbors are
    already calculated.

    Note that this function is designed to be used in the context of the SDR
    model. Because the algorithm is recursive upslope and downslope of each
    pixel, nodata values in the SDR input would propagate along the flow path.
    This case is not handled because we assume the SDR and flow dir inputs
    will come from the SDR model and have nodata in the same places.

    Args:
        mfd_flow_direction_path (string): a path to a raster with
            pygeoprocessing.routing MFD flow direction values.
        e_prime_path (string): path to a raster that shows sources of
            sediment that wash off a pixel but do not reach the stream.
        f_path (string): path to a raster that shows the sediment flux
            on a pixel for sediment that does not reach the stream.
        sdr_path (string): path to Sediment Delivery Ratio raster.
        target_sediment_deposition_path (string): path to created that
            shows where the E' sources end up across the landscape.

    Returns:
        None.

    """
    LOGGER.info('Calculate sediment deposition')
    cdef float target_nodata = -1
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_sediment_deposition_path,
        gdal.GDT_Float32, [target_nodata])
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, f_path,
        gdal.GDT_Float32, [target_nodata])

    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)
    cdef _ManagedRaster e_prime_raster = _ManagedRaster(
        e_prime_path, 1, False)
    cdef _ManagedRaster sdr_raster = _ManagedRaster(sdr_path, 1, False)
    cdef _ManagedRaster f_raster = _ManagedRaster(f_path, 1, True)
    cdef _ManagedRaster sediment_deposition_raster = _ManagedRaster(
        target_sediment_deposition_path, 1, True)

    # given the pixel neighbor numbering system
    #  3 2 1
    #  4 x 0
    #  5 6 7
    # if a pixel `x` has a neighbor `n` in position `i`,
    # then `n`'s neighbor in position `inflow_offsets[i]`
    # is the original pixel `x`
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_cols, n_rows = flow_dir_info['raster_size']
    cdef int mfd_nodata = 0
    cdef stack[int] processing_stack
    cdef float sdr_nodata = pygeoprocessing.get_raster_info(
        sdr_path)['nodata'][0]
    cdef float e_prime_nodata = pygeoprocessing.get_raster_info(
        e_prime_path)['nodata'][0]
    cdef int col_index, row_index, win_xsize, win_ysize, xoff, yoff
    cdef int global_col, global_row, flat_index, j, k
    cdef int seed_col = 0
    cdef int seed_row = 0
    cdef int neighbor_row, neighbor_col
    cdef int flow_val, neighbor_flow_val, ds_neighbor_flow_val
    cdef int flow_weight, neighbor_flow_weight
    cdef float flow_sum, neighbor_flow_sum
    cdef float downslope_sdr_weighted_sum, sdr_i, sdr_j
    cdef float p_j, p_val

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
            (xoff * yoff) / float(n_cols*n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index
                # check if this is a good seed pixel ( a local high point)
                if mfd_flow_direction_raster.get(seed_col, seed_row) == mfd_nodata:
                    continue
                seed_pixel = 1
                # iterate over each of the pixel's neighbors
                for j in range(8):
                    # skip if the neighbor is outside the raster bounds
                    neighbor_row = seed_row + ROW_OFFSETS[j]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        continue
                    neighbor_col = seed_col + COL_OFFSETS[j]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue
                    # skip if the neighbor's flow direction is undefined
                    neighbor_flow_val = <int>mfd_flow_direction_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_flow_val == mfd_nodata:
                        continue
                    # if the neighbor flows into it, it's not a local high
                    # point and so can't be a seed pixel
                    neighbor_flow_weight = (
                        neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                    if neighbor_flow_weight > 0:
                        seed_pixel = 0  # neighbor flows in, not a seed
                        break

                # if this can be a seed pixel and hasn't already been
                # calculated, put it on the stack
                if seed_pixel and sediment_deposition_raster.get(
                        seed_col, seed_row) == target_nodata:
                    processing_stack.push(seed_row * n_cols + seed_col)

                while processing_stack.size() > 0:
                    # loop invariant: cell has all upslope neighbors
                    # processed. this is true for seed pixels because they
                    # have no upslope neighbors.
                    flat_index = processing_stack.top()
                    processing_stack.pop()
                    global_row = flat_index // n_cols
                    global_col = flat_index % n_cols

                    # (sum over j ∈ J of f_j * p(i,j) in the equation for r_i)
                    # calculate the upslope f_j contribution to this pixel,
                    # the weighted sum of flux flowing onto this pixel from
                    # all neighbors
                    f_j_weighted_sum = 0
                    for j in range(8):
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue

                        # see if there's an inflow from the neighbor to the
                        # pixel
                        neighbor_flow_val = (
                            <int>mfd_flow_direction_raster.get(
                                neighbor_col, neighbor_row))
                        neighbor_flow_weight = (
                            neighbor_flow_val >> (inflow_offsets[j]*4)) & 0xF
                        if neighbor_flow_weight > 0:
                            f_j = f_raster.get(neighbor_col, neighbor_row)
                            if f_j == target_nodata:
                                continue
                            # sum up the neighbor's flow dir values in each
                            # direction.
                            # flow dir values are relative to the total
                            neighbor_flow_sum = 0
                            for k in range(8):
                                neighbor_flow_sum += (
                                    neighbor_flow_val >> (k*4)) & 0xF
                            # get the proportion of the neighbor's flow that
                            # flows into the original pixel
                            p_val = neighbor_flow_weight / neighbor_flow_sum
                            # add the neighbor's flux value, weighted by the
                            # flow proportion
                            f_j_weighted_sum += p_val * f_j

                    # calculate sum of SDR values of immediate downslope
                    # neighbors, weighted by proportion of flow into each
                    # neighbor
                    # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
                    downslope_sdr_weighted_sum = 0
                    flow_val = <int>mfd_flow_direction_raster.get(
                        global_col, global_row)
                    flow_sum = 0
                    for k in range(8):
                        flow_sum += (flow_val >> (k*4)) & 0xF

                    # iterate over the neighbors again
                    for j in range(8):
                        # skip if neighbor is outside the raster boundaries
                        neighbor_row = global_row + ROW_OFFSETS[j]
                        if neighbor_row < 0 or neighbor_row >= n_rows:
                            continue
                        neighbor_col = global_col + COL_OFFSETS[j]
                        if neighbor_col < 0 or neighbor_col >= n_cols:
                            continue
                        # if it is a downslope neighbor, add to the sum and
                        # check if it can be pushed onto the stack yet
                        flow_weight = (flow_val >> (j*4)) & 0xF
                        if flow_weight > 0:
                            sdr_j = sdr_raster.get(neighbor_col, neighbor_row)
                            if sdr_j == sdr_nodata:
                                continue
                            if sdr_j == 0:
                                # this means it's a stream, for SDR deposition
                                # purposes, we set sdr to 1 to indicate this
                                # is the last step on which to retain sediment
                                sdr_j = 1
                            p_j = flow_weight / flow_sum
                            downslope_sdr_weighted_sum += sdr_j * p_j

                            # check if we can add neighbor j to the stack yet
                            #
                            # if there is a downslope neighbor it
                            # couldn't have been pushed on the processing
                            # stack yet, because the upslope was just
                            # completed
                            upslope_neighbors_processed = 1
                            # iterate over each neighbor-of-neighbor
                            for k in range(8):
                                # no need to push the one we're currently
                                # calculating back onto the stack
                                if inflow_offsets[k] == j:
                                    continue
                                # skip if neighbor-of-neighbor is outside
                                # raster bounds
                                ds_neighbor_row = (
                                    neighbor_row + ROW_OFFSETS[k])
                                if ds_neighbor_row < 0 or ds_neighbor_row >= n_rows:
                                    continue
                                ds_neighbor_col = (
                                    neighbor_col + COL_OFFSETS[k])
                                if ds_neighbor_col < 0 or ds_neighbor_col >= n_cols:
                                    continue
                                # if any upslope neighbor of j hasn't been
                                # calculated, we can't push j onto the stack
                                # yet
                                ds_neighbor_flow_val = (
                                    <int>mfd_flow_direction_raster.get(
                                        ds_neighbor_col, ds_neighbor_row))
                                if (ds_neighbor_flow_val >> (
                                        inflow_offsets[k]*4)) & 0xF > 0:
                                    if (sediment_deposition_raster.get(
                                            ds_neighbor_col, ds_neighbor_row) ==
                                            target_nodata):
                                        upslope_neighbors_processed = 0
                                        break
                            # if all upslope neighbors of neighbor j are
                            # processed, we can push j onto the stack.
                            if upslope_neighbors_processed:
                                processing_stack.push(
                                    neighbor_row * n_cols +
                                    neighbor_col)

                    # nodata pixels should propagate to the results
                    sdr_i = sdr_raster.get(global_col, global_row)
                    if sdr_i == sdr_nodata:
                        continue
                    e_prime_i = e_prime_raster.get(global_col, global_row)
                    if e_prime_i == e_prime_nodata:
                        continue

                    if downslope_sdr_weighted_sum < sdr_i:
                        # i think this happens because of our low resolution
                        # flow direction, it's okay to zero out.
                        downslope_sdr_weighted_sum = sdr_i

                    # these correspond to the full equations for
                    # dr_i, r_i, and f_i given in the docstring
                    dr_i = (downslope_sdr_weighted_sum - sdr_i) / (1 - sdr_i)
                    r_i = dr_i * (e_prime_i + f_j_weighted_sum)
                    f_i = (1 - dr_i) * (e_prime_i + f_j_weighted_sum)

                    # On large flow paths, it's possible for r_i and f_i to
                    # have very small negative values that are numerically
                    # equivalent to 0. These negative values were raising
                    # questions on the forums and it's easier to clamp the
                    # values here than to explain IEEE 754.
                    if r_i < 0:
                        r_i = 0
                    if f_i < 0:
                        f_i = 0

                    sediment_deposition_raster.set(global_col, global_row, r_i)
                    f_raster.set(global_col, global_row, f_i)

    LOGGER.info('Sediment deposition 100% complete')
    sediment_deposition_raster.close()


def calculate_slope(dem_path_band, flow_accum_path_not_used, target_slope_path):
    cdef float slope_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        dem_path_band[0], target_slope_path,
        gdal.GDT_Float32, [slope_nodata], [slope_nodata])

    cdef _ManagedRaster dem_managed_raster = _ManagedRaster(
        dem_path_band[0], dem_path_band[1], False)
    cdef _ManagedRaster slope_managed_raster = _ManagedRaster(
        target_slope_path, 1, True)

    dem_info = pygeoprocessing.get_raster_info(dem_path_band[0])
    cdef int dem_nodata = dem_info['nodata'][0]
    cdef int n_cols, n_rows
    n_cols, n_rows = dem_info['raster_size']
    cdef float cellsize = abs(dem_info['pixel_size'][0])

    cdef float* neighbor_z = [0, 0, 0, 0, 0, 0, 0, 0]
    cdef float G, H
    cdef long n_pixels_visited = 0

    for offset_dict in pygeoprocessing.iterblocks(
            (target_slope_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Aspect %.2f%% complete', 100 * (
            n_pixels_visited / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index

                # Skip calculating aspect on this pixel if it's nodata.
                seed_elevation = dem_managed_raster.get(
                    seed_col, seed_row)
                if seed_elevation == dem_nodata:
                    continue

                # reset the neighbor_z list
                for i in range(8):
                    neighbor_z[i] = 0

                for neighbor_index in range(0, 8, 2):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row == -1 or neighbor_row == n_rows:
                        continue

                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col == -1 or neighbor_col == n_cols:
                        continue

                    neighbor_elevation = dem_managed_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_elevation == dem_nodata:
                        continue

                    neighbor_z[neighbor_index] = seed_elevation - neighbor_elevation

                G = (neighbor_z[0*2] - neighbor_z[2*2]) / (2.*cellsize)
                H = (neighbor_z[1*2] - neighbor_z[3*2]) / (2.*cellsize)

                slope_managed_raster.set(
                    seed_col, seed_row, cmath.atan(cmath.sqrt(G**2 + H**2)))

    slope_managed_raster.close()
    dem_managed_raster.close()



def calculate_dist_weighted_slope(
        dem_path_band, mfd_flow_accum_path_band, target_slope_path):
    cdef float slope_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        dem_path_band[0], target_slope_path,
        gdal.GDT_Float32, [slope_nodata], [slope_nodata])

    cdef _ManagedRaster dem_managed_raster = _ManagedRaster(
        dem_path_band[0], dem_path_band[1], False)
    cdef _ManagedRaster slope_managed_raster = _ManagedRaster(
        target_slope_path, 1, True)
    cdef _ManagedRaster mfd_flow_accum_managed_raster = _ManagedRaster(
        mfd_flow_accum_path_band[0], mfd_flow_accum_path_band[1], False)

    dem_info = pygeoprocessing.get_raster_info(dem_path_band[0])
    cdef int dem_nodata = dem_info['nodata'][0]
    cdef int n_cols, n_rows
    n_cols, n_rows = dem_info['raster_size']
    cdef float cellsize = abs(dem_info['pixel_size'][0])

    cdef float* neighbor_z = [0, 0, 0, 0, 0, 0, 0, 0]
    cdef float G, H
    cdef long n_pixels_visited = 0
    cdef float* neighbor_slopes = [0, 0, 0, 0, 0, 0, 0, 0]

    for offset_dict in pygeoprocessing.iterblocks(
            (target_slope_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Aspect %.2f%% complete', 100 * (
            n_pixels_visited / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index

                # Skip calculating aspect on this pixel if it's nodata.
                seed_elevation = dem_managed_raster.get(
                    seed_col, seed_row)
                if seed_elevation == dem_nodata:
                    continue

                seed_up_area = mfd_flow_accum_managed_raster.get(seed_col, seed_row)-1

                # reset the neighbor_z list
                for i in range(8):
                    neighbor_z[i] = 0

                for neighbor_index in range(0, 8, 2):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row == -1 or neighbor_row == n_rows:
                        continue

                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col == -1 or neighbor_col == n_cols:
                        continue

                    neighbor_elevation = dem_managed_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_elevation == dem_nodata:
                        continue

                    neighbor_z[neighbor_index] = seed_elevation - neighbor_elevation

                G = (neighbor_z[0*2] - neighbor_z[2*2]) / (2.*cellsize)
                H = (neighbor_z[1*2] - neighbor_z[3*2]) / (2.*cellsize)

                existing_slope = slope_managed_raster.get(seed_col, seed_row)
                if existing_slope == slope_nodata:
                    existing_slope = 0
                onseed_slope = cmath.atan(cmath.sqrt(G*G + H*H))
                upslope = existing_slope + cmath.log(seed_up_area) * onseed_slope

                slope_managed_raster.set(
                    seed_col, seed_row, upslope)

                neighbor_slopes_sum = 0
                for neighbor_index in range(8):
                    neighbor_slopes[neighbor_index] = 0
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        continue
                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue

                    # SAGA skips any neighbors that are level with or higher
                    # than the current pixel
                    neighbor_elevation = dem_managed_raster.get(
                        neighbor_col, neighbor_row)
                    d = seed_elevation - neighbor_elevation
                    if d <= 0:
                        continue

                    flow_length = FLOW_LENGTH[neighbor_index] * cellsize
                    neighbor_slope_dz = (d / flow_length)**1.1
                    neighbor_slopes[neighbor_index] = neighbor_slope_dz
                    neighbor_slopes_sum += neighbor_slope_dz

                for neighbor_index in range(8):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row < 0 or neighbor_row >= n_rows:
                        continue
                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col < 0 or neighbor_col >= n_cols:
                        continue

                    neighbor_slope = slope_managed_raster.get(neighbor_col, neighbor_row)
                    if neighbor_slope == slope_nodata:
                        neighbor_slope = 0

                    # avoid division by zero, falling back to neighbor_slope if the sum of neighbor slopes is 0.
                    if neighbor_slopes_sum > 0:
                        neighbor_slope += upslope * neighbor_slopes[neighbor_index] / neighbor_slopes_sum

                    slope_managed_raster.set(
                        neighbor_col, neighbor_row, neighbor_slope)

    slope_managed_raster.close()
    dem_managed_raster.close()




def calculate_aspect(dem_path, target_aspect_path, use_degrees=True):
    LOGGER.info('Calculating aspect')

    cdef float aspect_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        dem_path, target_aspect_path,
        gdal.GDT_Float32, [aspect_nodata], [aspect_nodata])

    cdef _ManagedRaster dem_managed_raster = _ManagedRaster(
        dem_path, 1, False)

    cdef _ManagedRaster aspect_managed_raster = _ManagedRaster(
        target_aspect_path, 1, True)

    dem_info = pygeoprocessing.get_raster_info(dem_path)
    cdef int dem_nodata = dem_info['nodata'][0]
    cdef int n_cols, n_rows
    n_cols, n_rows = dem_info['raster_size']
    cdef unsigned int n_pixels_visited = 0
    cdef int seed_row, seed_col, neighbor_row, neighbor_col

    cdef float* dzdx_factors = [2, 1, 0, -1, -2, -1, 0, 1]
    cdef float* dzdy_factors = [0, -1, -2, -1, 0, 1, 2, 1]
    for offset_dict in pygeoprocessing.iterblocks(
            (target_aspect_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Aspect %.2f%% complete', 100 * (
            n_pixels_visited / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index

                # Skip calculating aspect on this pixel if it's nodata.
                seed_elevation = dem_managed_raster.get(
                    seed_col, seed_row)
                if seed_elevation == dem_nodata:
                    continue

                # https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm#ESRI_SECTION1_4198691F8852475A9F4BC71246579FAA
                dz_n_pixels = 0
                dzdx = 0
                dzdy = 0
                for neighbor_index in range(8):
                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col == -1 or neighbor_col == n_cols:
                        continue

                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row == -1 or neighbor_row == n_rows:
                        continue

                    neighbor_elevation = dem_managed_raster.get(
                        neighbor_col, neighbor_row)
                    if neighbor_elevation == dem_nodata:
                        continue

                    dz_n_pixels += 1
                    dzdx += neighbor_elevation * dzdx_factors[neighbor_index]
                    dzdy += neighbor_elevation * dzdy_factors[neighbor_index]

                dzdx /= dz_n_pixels
                dzdy /= dz_n_pixels

                # 1 radian is 57.29578 degrees
                aspect_degrees = 57.29578 * cmath.atan2(dzdy, -1*dzdx)
                if aspect_degrees < 0:
                    aspect_degrees = 90 - aspect_degrees
                elif aspect_degrees > 90:
                    aspect_degrees = 360 - aspect_degrees + 90
                else:
                    aspect_degrees = 90 - aspect_degrees

                if use_degrees:
                    aspect = aspect_degrees
                else:
                    # convert to radians
                    aspect = aspect_degrees * (PI / 180.)

                aspect_managed_raster.set(
                    seed_col, seed_row, aspect)


    LOGGER.info('Aspect 100.00% complete')
    dem_managed_raster.close()
    aspect_managed_raster.close()


def calculate_average_aspect(
        mfd_flow_direction_path, target_average_aspect_path):
    """Calculate the Weighted Average Aspect Ratio from MFD.

    Calculates the average aspect ratio weighted by proportional flow
    direction.

    Args:
        mfd_flow_direction_path (string): The path to an MFD flow direction
            raster.
        target_average_aspect_path (string): The path to where the calculated
            weighted average aspect raster should be written.

    Returns:
        ``None``.

    """
    LOGGER.info('Calculating average aspect')

    cdef float average_aspect_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_average_aspect_path,
        gdal.GDT_Float32, [average_aspect_nodata], [average_aspect_nodata])

    flow_direction_info = pygeoprocessing.get_raster_info(
        mfd_flow_direction_path)
    cdef int mfd_flow_direction_nodata = flow_direction_info['nodata'][0]
    cdef int n_cols, n_rows
    n_cols, n_rows = flow_direction_info['raster_size']

    cdef _ManagedRaster mfd_flow_direction_raster = _ManagedRaster(
        mfd_flow_direction_path, 1, False)

    cdef _ManagedRaster average_aspect_raster = _ManagedRaster(
        target_average_aspect_path, 1, True)

    cdef int seed_row = 0
    cdef int seed_col = 0
    cdef int n_pixels_visited = 0
    cdef int win_xsize, win_ysize, xoff, yoff
    cdef int row_index, col_index, neighbor_index
    cdef int flow_weight_in_direction
    cdef int weight_sum
    cdef int seed_flow_value
    cdef float aspect_weighted_average, aspect_weighted_sum

    # the flow_lengths array is the functional equivalent
    # of calculating |sin(alpha)| + |cos(alpha)|.
    cdef float* flow_lengths = [
        1, <float>SQRT2,
        1, <float>SQRT2,
        1, <float>SQRT2,
        1, <float>SQRT2
    ]

    # Loop over iterblocks to maintain cache locality
    # Find each non-nodata pixel and calculate proportional flow
    # Multiply proportional flow times the flow length x_d
    # write the final value to the raster.
    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        LOGGER.info('Average aspect %.2f%% complete', 100 * (
            n_pixels_visited / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            seed_row = yoff + row_index
            for col_index in range(win_xsize):
                seed_col = xoff + col_index
                seed_flow_value = <int>mfd_flow_direction_raster.get(
                    seed_col, seed_row)

                # Skip this seed if it's nodata (Currently expected to be 0).
                # No need to set the nodata value here since we have already
                # filled the raster with nodata values at creation time.
                if seed_flow_value == mfd_flow_direction_nodata:
                    continue

                weight_sum = 0
                aspect_weighted_sum = 0
                for neighbor_index in range(8):
                    neighbor_row = seed_row + ROW_OFFSETS[neighbor_index]
                    if neighbor_row == -1 or neighbor_row == n_rows:
                        continue

                    neighbor_col = seed_col + COL_OFFSETS[neighbor_index]
                    if neighbor_col == -1 or neighbor_col == n_cols:
                        continue

                    flow_weight_in_direction = (seed_flow_value >> (
                        neighbor_index * 4) & 0xF)
                    weight_sum += flow_weight_in_direction

                    aspect_weighted_sum += (
                        flow_lengths[neighbor_index] *
                        flow_weight_in_direction)

                # Weight sum should never be less than 0.
                # Since it's an int, we can compare it directly against the
                # value of 0.
                if weight_sum == 0:
                    aspect_weighted_average = average_aspect_nodata
                else:
                    # We already know that weight_sum will be > 0 because we
                    # check for it in the condition above.
                    with cython.cdivision(True):
                        aspect_weighted_average = (
                            aspect_weighted_sum / <float>weight_sum)

                average_aspect_raster.set(
                    seed_col, seed_row, aspect_weighted_average)

        n_pixels_visited += win_xsize * win_ysize

    LOGGER.info('Average aspect 100.00% complete')

    mfd_flow_direction_raster.close()
    average_aspect_raster.close()



def flow_accumulation_mfd(
        flow_dir_mfd_raster_path_band, target_flow_accum_raster_path,
        avg_aspect_raster_path_band,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Multiple flow direction accumulation.

    Parameters:
        flow_dir_mfd_raster_path_band (tuple): a path, band number tuple
            for a multiple flow direction raster generated from a call to
            ``flow_dir_mfd``. The format of this raster is described in the
            docstring of that function.
        target_flow_accum_raster_path (str): a path to a raster created by
            a call to this function that is the same dimensions and projection
            as ``flow_dir_mfd_raster_path_band[0]``. The value in each pixel is
            1 plus the proportional contribution of all upstream pixels that
            flow into it. The proportion is determined as the value of the
            upstream flow dir pixel in the downslope direction pointing to
            the current pixel divided by the sum of all the flow weights
            exiting that pixel. Note the target type of this raster
            is a 64 bit float so there is minimal risk of overflow and the
            possibility of handling a float dtype in
            ``weight_raster_path_band``.
        avg_aspect_raster_path_band (tuple): as here in sdr_core.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow accumulation
            weight. If ``None``, 1 is the default flow accumulation weight.
            This raster must be the same dimensions as
            ``flow_dir_mfd_raster_path_band``. If a weight nodata pixel is
            encountered it will be treated as a weight value of 0.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.

    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.int32_t, ndim=2] flow_dir_mfd_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # These are used to estimate % complete
    cdef long long visit_count, pixel_count

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_n, yi_n, i_upstream_flow

    # used to hold flow direction values
    cdef int flow_dir_mfd, upstream_flow_weight

    # used as a holder variable to account for upstream flow
    cdef int compressed_upstream_flow_dir, upstream_flow_dir_sum

    # used to determine if the upstream pixel has been processed, and if not
    # to trigger a recursive uphill walk
    cdef double upstream_flow_accum

    cdef double flow_accum_nodata = IMPROBABLE_FLOAT_NODATA
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    # this value is used to store the current weight which might be 1 or
    # come from a predefined flow accumulation weight raster
    cdef double weight_val

    # `search_stack` is used to walk upstream to calculate flow accumulation
    # values represented in a flow pixel which stores the x/y position,
    # next direction to check, and running flow accumulation value.
    cdef stack[FlowPixelType] search_stack
    cdef FlowPixelType flow_pixel

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    if not _is_raster_path_band_formatted(flow_dir_mfd_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                flow_dir_mfd_raster_path_band))
    if weight_raster_path_band and not _is_raster_path_band_formatted(
            weight_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                weight_raster_path_band))

    LOGGER.debug('creating target flow accum raster layer')
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0], target_flow_accum_raster_path,
        gdal.GDT_Float64, [flow_accum_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    flow_accum_managed_raster = _ManagedRaster(
        target_flow_accum_raster_path, 1, 1)

    avg_aspect_managed_raster = _ManagedRaster(
        avg_aspect_raster_path_band[0], avg_aspect_raster_path_band[1], False)

    # make a temporary raster to mark where we have visisted
    LOGGER.debug('creating visited raster layer')
    tmp_dir_root = os.path.dirname(target_flow_accum_raster_path)
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_root, prefix='mfd_flow_dir_')
    visited_raster_path = os.path.join(tmp_dir, 'visited.tif')
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0], visited_raster_path,
        gdal.GDT_Byte, [0],
        raster_driver_creation_tuple=('GTiff', (
            'SPARSE_OK=TRUE', 'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS))))
    visited_managed_raster = _ManagedRaster(visited_raster_path, 1, 1)

    flow_dir_managed_raster = _ManagedRaster(
        flow_dir_mfd_raster_path_band[0], flow_dir_mfd_raster_path_band[1], 0)
    flow_dir_raster = gdal.OpenEx(
        flow_dir_mfd_raster_path_band[0], gdal.OF_RASTER)
    flow_dir_band = flow_dir_raster.GetRasterBand(
        flow_dir_mfd_raster_path_band[1])

    cdef _ManagedRaster weight_raster = None
    if weight_raster_path_band:
        weight_raster = _ManagedRaster(
            weight_raster_path_band[0], weight_raster_path_band[1], 0)
        raw_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if raw_weight_nodata is not None:
            weight_nodata = raw_weight_nodata

    flow_dir_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_mfd_raster_path_band[0])
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']
    pixel_count = raster_x_size * raster_y_size
    visit_count = 0

    cdef float cell_size = cmath.fabs(flow_dir_raster_info['cell_size'])

    LOGGER.debug('starting search')
    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_mfd_raster_path_band, offset_only=True,
            largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        flow_dir_mfd_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.int32)
        flow_dir_mfd_buffer_array[:] = 0  # 0 means no flow at all

        # check if we can widen the border to include real data from the
        # raster
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        flow_dir_mfd_buffer_array[ya:yb, xa:xb] = flow_dir_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.int32)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to set flow accumulation
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                flow_dir_mfd = flow_dir_mfd_buffer_array[yi, xi]
                if flow_dir_mfd == 0:
                    # no flow in this pixel, so skip
                    continue

                for i_n in range(8):
                    if ((flow_dir_mfd >> (i_n * 4)) & 0xF) == 0:
                        # no flow in that direction
                        continue
                    xi_n = xi+D8_XOFFSET[i_n]
                    yi_n = yi+D8_YOFFSET[i_n]

                    if flow_dir_mfd_buffer_array[yi_n, xi_n] == 0:
                        # if the entire value is zero, it flows nowhere
                        # and the root pixel is draining to it, thus the
                        # root must be a drain
                        xi_root = xi-1+xoff
                        yi_root = yi-1+yoff
                        if weight_raster is not None:
                            weight_val = <double>weight_raster.get(
                                xi_root, yi_root)
                            if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                weight_val = 0.0
                        else:
                            weight_val = 1.0
                        search_stack.push(
                            FlowPixelType(xi_root, yi_root, 0, weight_val))
                        visited_managed_raster.set(xi_root, yi_root, 1)
                        visit_count += 1
                        break

                while not search_stack.empty():
                    flow_pixel = search_stack.top()
                    search_stack.pop()

                    if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
                        last_log_time = ctime(NULL)
                        LOGGER.info(
                            'mfd flow accum %.1f%% complete',
                            100.0 * visit_count / float(pixel_count))

                    preempted = 0
                    for i_n in range(flow_pixel.last_flow_dir, 8):
                        xi_n = flow_pixel.xi+D8_XOFFSET[i_n]
                        yi_n = flow_pixel.yi+D8_YOFFSET[i_n]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # no upstream here
                            continue
                        compressed_upstream_flow_dir = (
                            <int>flow_dir_managed_raster.get(xi_n, yi_n))
                        upstream_flow_weight = (
                            compressed_upstream_flow_dir >> (
                                D8_REVERSE_DIRECTION[i_n] * 4)) & 0xF
                        if upstream_flow_weight == 0:
                            # no upstream flow to this pixel
                            continue
                        upstream_flow_accum = (
                            flow_accum_managed_raster.get(xi_n, yi_n))
                        if (_is_close(upstream_flow_accum, flow_accum_nodata, 1e-8, 1e-5)
                                and not visited_managed_raster.get(
                                    xi_n, yi_n)):
                            # process upstream before this one
                            flow_pixel.last_flow_dir = i_n
                            search_stack.push(flow_pixel)
                            if weight_raster is not None:
                                weight_val = <double>weight_raster.get(
                                    xi_n, yi_n)
                                if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                    weight_val = 0.0
                            else:
                                weight_val = 1.0
                            search_stack.push(
                                FlowPixelType(xi_n, yi_n, 0, weight_val))
                            visited_managed_raster.set(xi_n, yi_n, 1)
                            visit_count += 1
                            preempted = 1
                            break
                        upstream_flow_dir_sum = 0
                        for i_upstream_flow in range(8):
                            upstream_flow_dir_sum += (
                                compressed_upstream_flow_dir >> (
                                    i_upstream_flow * 4)) & 0xF

                        flow_pixel.value += (
                            upstream_flow_accum * upstream_flow_weight /
                            <float>upstream_flow_dir_sum)
                    if not preempted:
                        flow_pixel.value /= cell_size * avg_aspect_managed_raster.get(
                            flow_pixel.xi, flow_pixel.yi)
                        flow_accum_managed_raster.set(
                            flow_pixel.xi, flow_pixel.yi,
                            flow_pixel.value)
    flow_accum_managed_raster.close()
    flow_dir_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    visited_managed_raster.close()
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        LOGGER.exception("couldn't remove temp dir")
    LOGGER.info('%.1f%% complete', 100.0)

