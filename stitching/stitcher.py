import logging

import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress
from skimage.feature import register_translation
import dask.array as da

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
            logger.warning("collection not fused yet, please call `Stitcher.fuse()`")
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

    def adjust_intensity(self):
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

    def fuse(self, dst_dir, chunk_shape, compressor=None, overwrite=False):
        vol_shape = self.collection.bounding_box

        # create zarr directory store
        mode = "w" if overwrite else "w-"
        vol = zarr.open(
            dst_dir,
            mode=mode,
            shape=vol_shape,
            chunks=chunk_shape,  # FIXME auto determine chunk shape
            dtype=np.uint16,
            compressor=compressor,
        )

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

        # paste tiles into vol
        vol = da.from_zarr(vol)
        for tile in self.collection.tiles:
            self._fuse_tile(vol, chunk_shape, tile)

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
        nn_tiles = self._select_pos_neighbors(ref_tile)
        afit = []
        bfit = []
        for nn_tile in nn_tiles:
            ref_roi, ref_raw = ref_tile.overlap_roi(nn_tile, return_raw_roi=True)
            nn_roi, nn_raw = nn_tile.overlap_roi(ref_tile, return_raw_roi=True)
            if (ref_raw is None) or (nn_raw is None):
                slope, intercept = 1.0, 0.0
            else:
                ref_raw = np.ravel(ref_raw)
                nn_raw = np.ravel(nn_raw)
                slope, intercept, r_value, p_value, std = linregress(nn_raw, ref_raw)
            afit.append(slope)
            bfit.append(intercept)
        nnfit = {"afit": afit, "bfit": bfit}
        return nnfit

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

    def _fuse_tile(self, vol, chunk_shape, tile):
        data = tile.data
        dshape = data.shape
        coord0 = tile.coord
        logger.debug(
            f"fuse tile: index={tile.index}, dshape={dshape}, chunk_shape={chunk_shape}, coord={coord0}"
        )

        def _resample_block(block, block_info=None):
            print(block_info)
            print()

        vol.map_blocks(_resample_block)

        # Partition the whole tile into chunks.
        chunk_mesh = np.meshgrid(
            *tuple(
                np.arange(x, x + L, Lc) for x, L, Lc in zip(coord0, dshape, chunk_shape)
            )
        )
        chunks_x0 = [pt for pt in zip(*(x.flat for x in chunk_mesh))]

        for chunk_x0 in chunks_x0:
            cx0 = tuple((np.array(chunk_x0) - np.array(coord0)).astype(int))
            cshape = [
                L - xc if xc + Lc > L else Lc
                for xc, Lc, L in zip(cx0, chunk_shape, dshape)
            ]
            axes_mesh0 = [
                np.linspace(xc, xc + Lc - 1, Lc) for xc, Lc in zip(chunk_x0, cshape)
            ]
            if len(cshape) == 2:
                chunk_data = data[
                    cx0[0] : cx0[0] + cshape[0], cx0[1] : cx0[1] + cshape[1]
                ]
            else:
                chunk_data = data[
                    cx0[0] : cx0[0] + cshape[0],
                    cx0[1] : cx0[1] + cshape[1],
                    cx0[2] : cx0[2] + cshape[2],
                ]
            fusefunc = RegularGridInterpolator(
                axes_mesh0, chunk_data, bounds_error=False, fill_value=None
            )

            chunk_x1 = tuple(np.round(x) for x in chunk_x0)
            cx1 = tuple(np.array(chunk_x1).astype(int))
            mesh1 = np.meshgrid(
                *tuple(np.linspace(x, x + L - 1, L) for x, L in zip(chunk_x1, cshape)),
                indexing="ij",
            )
            pts1 = fusefunc([pt for pt in zip(*(x.flat for x in mesh1))])
            pts1[pts1 < 0] = 0
            pts1 = np.round(pts1).astype(np.uint16).reshape(cshape)
            if len(cshape) == 2:
                vol[cx1[0] : cx1[0] + cshape[0], cx1[1] : cx1[1] + cshape[1]] = pts1
            else:
                vol[
                    cx1[0] : cx1[0] + cshape[0],
                    cx1[1] : cx1[1] + cshape[1],
                    cx1[2] : cx1[2] + cshape[2],
                ] = pts1

    ##
