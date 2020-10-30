import logging

import dask.array as da
import numpy as np
import zarr
from dask import delayed
from dask.delayed import Delayed
from dask.distributed import Client, Future, as_completed
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress
from skimage.feature import register_translation

from stitching.tiles import Tile

__all__ = ["Stitcher"]

logger = logging.getLogger(__name__)


class Stitcher(object):
    def __init__(self, collection):
        self._collection = collection
        self._fused = None

    ##

    @property
    def collection(self):
        return self._collection

    @property
    def fused(self):
        if self._fused is None:
            logger.error("collection not fused yet, please call `Stitcher.fuse()`")
        return self._fused

    ##

    def align(self):
        for ref_tile in self.collection.tiles:
            nn_tiles = self.collection.neighbor_of(ref_tile)

            print(str(ref_tile))
            for nn_tile in nn_tiles:
                print(f".. {str(nn_tile)}")

                ref_roi = ref_tile.overlap_roi(nn_tile)
                nn_roi = nn_tile.overlap_roi(ref_tile)

                if (ref_roi is not None) and (nn_roi is not None):
                    shift, error, _ = register_translation(
                        ref_roi, nn_roi, upsample_factor=1, return_error=True
                    )
                    print(f"shifts:{shift}, error:{error:04f}")
                    nn_tile.shift(shift)

    def adjust_intensity(self):  # TODO deprecate this
        logger.info(f"estimate global histogram")

        # estimate histogram min/max
        m, M = None, None
        for tile in self.collection.tiles:
            _m = tile.data.min()
            try:
                if _m < m:
                    m = _m
            except TypeError:
                m = _m

            _M = tile.data.max()
            try:
                if _M > M:
                    M = _M
            except TypeError:
                M = _M
        logger.debug(f"global: min={m}, max={M}")

        # collect histograms
        bins = np.linspace(m, M, 256)
        h = None
        for tile in self.collection.tiles:
            _h, _ = np.histogram(tile.data, bins=bins)
            try:
                h += _h
            except TypeError:
                h = _h

        n_pixels = np.sum(h)
        assert n_pixels > 0, "number of pixels should be greater than 0"
        limit, threshold = n_pixels / 10, n_pixels / 5000

        # find min
        for _h, e in zip(h, bins):
            if _h > limit:
                continue
            elif _h > threshold:
                m = e
                break
        # find max
        for _h, e in zip(h[::-1], bins[::-1]):
            if _h > limit:
                continue
            elif _h > threshold:
                M = e
                break
        min_max = (m, M)
        logger.info(f"auto-threshold {min_max}")

        # update intensities
        for tile in self.collection.tiles:
            tile.handle.setLevels(min_max)

    def fuse(self, dst_dir, overwrite=False):
        vol_shape = self.collection.bounding_box
        first_tile_index = sorted(list(self.collection.layout.indices))[0]
        first_tile = self.collection[first_tile_index]

        # create zarr directory store
        mode = "w" if overwrite else "w-"
        # TODO add compression
        dst_arr = zarr.open(
            dst_dir,
            mode=mode,
            shape=vol_shape,
            dtype=first_tile.data.dtype
        )

        """
        # DEBUG -> deprecate the following sections
        self._estimate_global_intensity_profile()

        pxlsts = {}
        for tile in self.collection.tiles:
            idx = tile.index
            npxl = tile.data.size
            pxlmean = np.mean(tile.data)
            pxlstd = np.std(tile.data)
            # TODO replace sum/ssum with simple np.sum(*)
            pxlsum = pxlmean * npxl
            pxlssum = (pxlstd ** 2 + pxlmean ** 2) * npxl
            nnfit = self._fit_intensity_leastsq(tile)
            t_pxlsts = {
                "npxl": npxl,
                "pxlmean": pxlmean,
                "pxlstd": pxlstd,
                "pxlsum": pxlsum,
                "pxlssum": pxlssum,
                "pxladj_a": 1.0,
                "pxladj_b": 0.0,
                "nnfit": nnfit,
            }
            pxlsts[idx] = t_pxlsts

        # adjust tile pixels sequentially starting from the first tile
        first_tile_index = sorted(list(self.collection.layout.indices))[0]
        first_tile = self.collection[first_tile_index]
        self._fuse_para_adjust(first_tile, 0, pxlsts)
        self._fuse_pxl_adjust(pxlsts)
        """

        # paste tiles into vol
        for tile in self.collection.tiles:
            logger.info(f"writing tile {tile.index}")
            self._fuse_tile(dst_arr, tile)

    def _select_pos_neighbors(self, ref_tile):  # QA
        """
        Select index-positive neighbors of a reference tile.

        Args:
            ref_tile (Tile): the reference tile

        Note:
            If stitching order is assigned, use this function to reorder how neighbors 
            are assigned.
        """
        ref_index = ref_tile.index
        nn_tiles = []
        for nn_tile in self.collection.neighbor_of(ref_tile):
            nn_index = nn_tile.index
            if all(i >= j for i, j in zip(nn_index, ref_index)):
                nn_tiles.append(nn_tile)
        return nn_tiles

    def _fit_intensity_leastsq(self, ref_tile):
        """
        Fit intensity between overlap regions with a linear function using least square 
        method.

        Y = m X + c
            Y: overlap region of the reference
            X: overlap region of the source (neighbor)
        
        Args:
            ref_tile (Tile): the reference tile
        """
        slopes, intercepts = [], []
        for nn_tile in self._select_pos_neighbors(ref_tile):
            m, c = ref_tile.match_intensity(nn_tile, apply=False)
            slopes.append(m)
            intercepts.append(c)
        nnfit = {"afit": slopes, "bfit": intercepts}
        return nnfit

    def _estimate_global_intensity_profile(self):
        """
        Walk over all tiles and collect their statistics, and estimate a proper
        intensity correction function for each tile.
        """
        # TODO start tile should not assume to be origin tile
        first_tile_index = sorted(list(self.collection.layout.indices))[0]
        start_tile = self.collection[first_tile_index]

        client = Client.current()

        class TileStatistics:
            """
            A aggregator for tile statistics and their task management.
            """

            def __init__(self, tile: Tile):
                # alias for the internal data container
                self.data = tile.data

                # populate all delay functions
                self._std = self.data.std()
                self._sum = self.data.sum()
                self._ssum = (self.data * self.data).sum()

            ##

            @property
            def size(self):
                """Number of elements in the array."""
                return self.data.size

            @property
            def mean(self):
                """
                Compute the arithmetic mean.
                
                Manually calculate mean instead of calling .mean() function to ensure 
                we use the cached version.
                """
                return self.sum / self.size

            @property
            def std(self):
                """Sample standard deviation of array elements."""
                assert not isinstance(self._std, Delayed), "statistics not computed"

                if isinstance(self._std, Future):
                    self._std = self._std.result()
                return self._std

            @property
            def sum(self):
                """Sum of array elements."""
                assert not isinstance(self._sum, Delayed), "statistics not computed"

                if isinstance(self._sum, Future):
                    self._sum = self._sum.result()
                return self._sum

            @property
            def ssum(self):
                """Squared sum of array elements."""
                assert not isinstance(self._ssum, Delayed), "statistics not computed"

                if isinstance(self._ssum, Future):
                    self._ssum = self._ssum.result()
                return self._ssum

            ##

            def compute(self):
                """
                Trigger compute on delayed functions.

                Args:
                    client (Client, optional): the scheduler to operate on

                Returns:
                    (list of Future): the Future object to wait for
                """
                self._std = client.compute(self._std)
                self._sum = client.compute(self._sum)
                self._ssum = client.compute(self._ssum)

                return [self._std, self._sum, self._ssum]

        # build statistics collection...
        statistics = {tile: TileStatistics(tile) for tile in self.collection.tiles}
        # ... and wait for them to finish
        batch_size = len(client.ncores())
        tasks = list(statistics.values())
        n_failed = 0
        for i in range(0, len(tasks), batch_size):
            futures = []
            for task in tasks[i : i + batch_size]:
                futures.extend(task.compute())
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    n_failed += 1
                    logger.exception(f"{n_failed} failed task(s)")

        # DFS over all the tiles
        visited = {tile: False for tile in self.collection.tiles}
        stack = [start_tile]
        while stack:
            curr_tile = stack.pop()
            if visited[curr_tile]:
                continue
            visited[curr_tile] = True

            # TODO process shits

            # next iteration
            for nn_tile in self._select_pos_neighbors(curr_tile):
                stack.append(nn_tile)

    def _fuse_para_adjust(self, ref_tile, idir, pxlsts):
        maxdir = len(self.collection.layout.tile_shape) - 1
        ref_pxlsts = pxlsts[ref_tile.index]
        nn_tiles = self._select_pos_neighbors(ref_tile)

        while len(nn_tiles) > 0:
            # myself pixel-adjust parameter
            afit0 = ref_pxlsts["pxladj_a"]
            bfit0 = ref_pxlsts["pxladj_b"]
            # next tile index along idir direction
            next_tile = None
            next_tidx = tuple(
                x + 1 if i == idir else x for i, x in enumerate(ref_tile.index)
            )
            if idir < maxdir:
                self._fuse_para_adjust(ref_tile, idir + 1, pxlsts)

            for ii, nn_tile in enumerate(nn_tiles):
                # adjust pixel-adjust parameters of neighboring tiles
                afit = ref_pxlsts["nnfit"]["afit"][ii]
                bfit = ref_pxlsts["nnfit"]["bfit"][ii]
                ref_pxlsts["nnfit"]["afit"][ii] = afit * afit0
                ref_pxlsts["nnfit"]["bfit"][ii] = afit * bfit0 + bfit
                # adjust pixel sum & ssum of neighboring along idir direction
                if nn_tile.index == next_tidx:
                    logger.debug(f"adjust parameter {next_tidx}")

                    next_tile = nn_tile
                    next_pxlsts = pxlsts[next_tidx]
                    npxl = next_pxlsts["npxl"]
                    pxls = next_pxlsts["pxlsum"]
                    pxlss = next_pxlsts["pxlssum"]
                    afit = ref_pxlsts["nnfit"]["afit"][ii]
                    bfit = ref_pxlsts["nnfit"]["bfit"][ii]
                    nn_pxlsum = afit * pxls + npxl * bfit
                    nn_pxlssum = (
                        afit ** 2 * pxlss + 2.0 * afit * bfit * pxls + npxl * bfit ** 2
                    )
                    pxlsts[next_tidx]["pxladj_a"] = afit
                    pxlsts[next_tidx]["pxladj_b"] = bfit
                    pxlsts[next_tidx]["pxlsum"] = nn_pxlsum
                    pxlsts[next_tidx]["pxlssum"] = nn_pxlssum
                    pxlsts[next_tidx]["pxlmean"] = nn_pxlsum / npxl
                    pxlsts[next_tidx]["pxlstd"] = np.sqrt(
                        nn_pxlssum / npxl - (nn_pxlsum / npxl) ** 2
                    )
            if next_tile == None:
                break
            ref_tile = next_tile
            nn_tiles = self._select_pos_neighbors(ref_tile)

    def _fuse_pxl_adjust(self, pxlsts):
        tiles = self.collection.tiles
        pxls_mean = []
        pxls_std = []
        pxlsum0 = 0.0
        pxlssum0 = 0.0
        npxl0 = 0
        for tile in tiles:
            t_pxlsts = pxlsts[tile.index]
            npxl = t_pxlsts["npxl"]
            npxl0 += npxl
            pxlsum0 += t_pxlsts["pxlsum"]
            pxlssum0 += t_pxlsts["pxlssum"]
            pxls_mean.append(t_pxlsts["pxlmean"])
            pxls_std.append(t_pxlsts["pxlstd"])
        # total mean and std of the current whole volume.
        pxlmean0 = pxlsum0 / npxl0
        pxlstd0 = np.sqrt(pxlssum0 / npxl0 - pxlmean0 ** 2)
        # the target mean and std to adjust of the whole volume.
        pxlmean1 = np.median(np.asarray(pxls_mean))
        pxlstd1 = np.amax(np.asarray(pxls_std))

        for tile in tiles:
            logger.debug(f"adjust tile pixels {tile.index}")
            t_pxlsts = pxlsts[tile.index]
            afit = t_pxlsts["pxladj_a"]
            bfit = t_pxlsts["pxladj_b"]
            tile._data = np.round(
                (afit * tile.data.astype(np.float32) + bfit - pxlmean0)
                / pxlstd0
                * pxlstd1
                + pxlmean1
            )
            tile._data = np.where(tile.data < 0, 0, tile.data).astype(np.uint16)

    def _resample_tile(self, src_tile: Tile):
        """
        Resample tile to destination grid.

        Args:
            tile (Tile): tile to resample with

        Returns:
            TBA
        """
        data = src_tile.data

        # offset in local coordinate \in (-1, 1)
        offset, coord0 = np.modf(src_tile.coord)
        coord0 = coord0.astype(int)  # indexing is always integer
        logger.debug(
            f"tile {src_tile.index}, coord0={tuple(coord0)}, offset={tuple(offset)}"
        )

        def _extract_result_shape(block_info, pad_size=1):
            # NOTE range \in [start, stop)
            array_loc = block_info[0]["array-location"]
            return tuple((stop - start) - 2 * pad_size for start, stop in array_loc)

        def _build_mesh_vector(block_info):
            array_loc = block_info[0]["array-location"]
            return tuple(
                np.arange(start, stop, dtype=np.float32) for start, stop in array_loc
            )

        def resample_block(block, block_info=None):
            # infer actual block size
            result_shape = _extract_result_shape(block_info)

            # generate orignal mesh
            vec0 = _build_mesh_vector(block_info)
            interp = RegularGridInterpolator(vec0, block)

            # generate target mesh, overlap region is 1 voxel
            vec = tuple(v[1:-1] + o for v, o in zip(vec0, offset))
            mesh = np.meshgrid(*vec, indexing="ij")

            # convert to lookup table (list of points)
            ndim = len(mesh)
            mesh = np.array(mesh).reshape(ndim, -1)

            result = interp(mesh.T)

            # restore shape
            result = result.reshape(result_shape)

            return result

        res_tile = da.map_overlap(
            data,
            resample_block,
            dtype=data.dtype,
            depth=1,
            boundary="nearest",
            chunks=data.chunks,
            trim=False,
        )

        return coord0, res_tile

    def _fuse_tile(self, dst_arr, tile: Tile):
        # TODO rewrite this to factor in overlap regions

        origin = self.collection.origin
        coord0, res_tile = self._resample_tile(tile)
        origin, shape = np.array(origin), np.array(tile.data.shape)
        dtype = dst_arr.dtype

        # build sampler
        sampler_start = coord0 - origin
        sampler_stop = sampler_start + shape
        sampler = tuple(
            slice(start, stop) for start, stop in zip(sampler_start, sampler_stop)
        )
        dst_arr[sampler] = res_tile.compute().astype(dtype)

#    def _fuse_tile_0(self, dst_arr, tile: Tile):
#        chunk_shape = tile.data.chunksize
#
#        data = tile.data
#        dshape = data.shape
#        coord0 = tile.coord
#        logger.debug(
#            f"fuse (tile): index={tile.index}, dshape={dshape}, chunk_shape={chunk_shape}, coord={coord0}"
#        )
#
#        # Partition the whole tile into chunks.
#        chunk_mesh = np.meshgrid(
#            *tuple(
#                np.arange(x, x + L, Lc) for x, L, Lc in zip(coord0, dshape, chunk_shape)
#            )
#        )
#        chunks_x0 = [pt for pt in zip(*(x.flat for x in chunk_mesh))]
#
#        for chunk_x0 in chunks_x0:
#            cx0 = tuple((np.array(chunk_x0) - np.array(coord0)).astype(int))
#            cshape = [
#                L - xc if xc + Lc > L else Lc
#                for xc, Lc, L in zip(cx0, chunk_shape, dshape)
#            ]
#            axes_mesh0 = [
#                np.linspace(xc, xc + Lc - 1, Lc) for xc, Lc in zip(chunk_x0, cshape)
#            ]
#            if len(cshape) == 2:
#                chunk_data = data[
#                    cx0[0] : cx0[0] + cshape[0], cx0[1] : cx0[1] + cshape[1]
#                ]
#            else:
#                chunk_data = data[
#                    cx0[0] : cx0[0] + cshape[0],
#                    cx0[1] : cx0[1] + cshape[1],
#                    cx0[2] : cx0[2] + cshape[2],
#                ]
#            fusefunc = RegularGridInterpolator(
#                axes_mesh0, chunk_data, bounds_error=False, fill_value=None
#            )
#
#            chunk_x1 = tuple(np.round(x) for x in chunk_x0)
#            cx1 = tuple(np.array(chunk_x1).astype(int))
#            mesh1 = np.meshgrid(
#                *tuple(np.linspace(x, x + L - 1, L) for x, L in zip(chunk_x1, cshape)),
#                indexing="ij",
#            )
#            pts1 = fusefunc([pt for pt in zip(*(x.flat for x in mesh1))])
#            pts1[pts1 < 0] = 0
#            pts1 = np.round(pts1).astype(np.uint16).reshape(cshape)
#            if len(cshape) == 2:
#                dst_arr[cx1[0] : cx1[0] + cshape[0], cx1[1] : cx1[1] + cshape[1]] = pts1
#            else:
#                dst_arr[
#                    cx1[0] : cx1[0] + cshape[0],
#                    cx1[1] : cx1[1] + cshape[1],
#                    cx1[2] : cx1[2] + cshape[2],
#                ] = pts1
#
    ##
