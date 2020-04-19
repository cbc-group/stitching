import logging

import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator
from skimage.feature import register_translation

__all__ = ["Stitcher"]

logger = logging.getLogger(__name__)


class Stitcher(object):
    def __init__(self, collection):
        self._collection = collection

    ##

    @property
    def collection(self):
        return self._collection

    ##

    def align(self):
        for ref_tile in self.collection.tiles:
            nn_tiles = self.collection.neighbor_of(ref_tile)

            print(str(ref_tile))
            for nn_tile in nn_tiles:
                print(f".. {str(nn_tile)}")

                ref_roi = ref_tile.overlap_roi(nn_tile)
                nn_roi = nn_tile.overlap_roi(ref_tile)

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

    def fuse(self, outdir, chunk_shape, compressor):
        # get the shape of the whole volume.
        tile_shape = self.collection.layout.tile_shape
        last_tidx = tuple(i-1 for i in tile_shape)
        last_tile = self.collection._tiles[last_tidx]
        vol_shape = tuple(np.round(last_tile.coord).astype(int) + last_tile.data.shape)
        vol = zarr.open(
            outdir,
            mode="w",
            shape=vol_shape,
            chunks=chunk_shape,
            dtype="u2",
            compressor=compressor
        )

        #
        # handling overlapped regions of tiles with their neighbors.
        #

        # paste tiles into vol.
        tiles = self.collection.tiles
        for tile in tiles:
            _fuse_tile(vol, tile)

    def _fuse_tile(vol, tile):
        data = tile.data
        dshape = data.shape
        coord0 = tile.coord
        axes_mesh0 = tuple(np.linspace(x,x+L-1,L) for x, L in zip(coord0, dshape))
        fusefunc = RegularGridInterpolator(axes_mesh0, data, bounds_error=False, fill_value=None)

        coord1 = tuple(np.round(x).astype(int) for x in coord0)
        mesh1 = np.meshgrid(*tuple(np.linspace(x,x+L-1,L) for x, L in zip(coord1, dshape)), indexing='ij')
        pts1 = [ pt for pt in zip(*(x.flat for x in mesh1)) ]
        pxls1 = fusefunc(pts1).reshape(dshape)
        if (len(dshape) == 2):
            vol[coord1[0]:coord1[0]+dshape[0],
                coord1[1]:coord1[1]+dshape[1]] = pxls1
        else:
            vol[coord1[0]:coord1[0]+dshape[0],
                coord1[1]:coord1[1]+dshape[1],
                coord1[2]:coord1[2]+dshape[2]] = pxls1
    ##
