from collections import defaultdict
from dataclasses import dataclass
import glob
from itertools import product
import logging
from math import ceil, floor
import os
from typing import Tuple

import imageio
import numpy as np
import pandas as pd
import pyqtgraph as pg
from skimage.exposure import histogram, match_histograms
from skimage.feature import register_translation

from utils import find_dataset_dir

__all__ = []

logger = logging.getLogger(__name__)


@dataclass
class Tile(object):
    index: Tuple[int]
    coord: Tuple[float]
    data: np.ndarray
    display: pg.ImageItem = None

    def __hash__(self):
        return hash(self.index)

    def __str__(self):
        return f"<Tile, {self.data.shape}, {self.data.dtype} @ {self.index}>"


class Stitcher(object):
    def __init__(self, coords, data):
        # convert real scale to rank
        ranks = coords.rank(axis="index", method="dense")
        ranks = ranks.astype(int) - 1

        # link data with index and coordinate
        self._tiles = [
            Tile(tuple(r.values[::-1]), tuple(c.values[::-1]), d)
            for (_, r), (_, c), d in zip(ranks.iterrows(), coords.iterrows(), data)
        ]

        self._tile_shape = coords.nunique().values[::-1]
        desc = ", ".join([f"{a}:{n}" for a, n in zip("xyz", self.tile_shape[::-1])])
        logger.info(f"tile shape ({desc})")

    ##

    @property
    def tiles(self):
        return self._tiles

    @property
    def tile_shape(self):
        return self._tile_shape

    ##

    def align(self):
        app = pg.mkQApp()

        window = pg.GraphicsLayoutWidget()
        window.setWindowTitle("Preview Alignment")

        vb = window.addViewBox()
        vb.setAspectLocked()
        vb.invertY()
        vb.enableAutoRange()

        window.show()

        # adjust intensities
        global_hist = defaultdict(lambda: 0)
        logger.info("match neighboring histograms")
        for tile in self.tiles:
            print(tile.index)

            values, counts = np.unique(tile.data.ravel(), return_counts=True)
            for v, c in zip(values, counts):
                global_hist[v] += c
        # extract result histogram
        template_values, counts = tuple(zip(*list(sorted(global_hist.items()))))
        template_values, counts = np.array(template_values), np.array(counts)
        template_quantiles = np.cumsum(counts) / counts.sum()

        logger.debug(f"{len(template_values)} unique values")
        # adjust everyone in current tile
        logger.info("remap to global cdf")
        for tile in self.tiles:
            print(tile.index)
            v, i, c = np.unique(
                tile.data.ravel(), return_inverse=True, return_counts=True
            )
            q = np.cumsum(c) / tile.data.size

            # interp to target histogram
            iv = np.interp(q, template_quantiles, template_values)

            # restore to correct size
            tile.data = iv[i].reshape(tile.data.shape)

        # montage
        logger.info("generate montage")
        for tile in self.tiles:
            print(tile.index)

            # create display
            image = pg.ImageItem(tile.data)
            image.setOpts(axisOrder="row-major")
            vb.addItem(image)
            tile.display = image

            # shift to default position
            # TODO assuming it is 2d
            coord = [c * n for c, n in zip(tile.index[1:][::-1], tile.data.shape)]
            image.setPos(pg.Point(*coord))

            # force screen update
            pg.QtGui.QApplication.processEvents()

        # generate neighbor list
        neighbors = self._list_neighbors()

        
        logger.info("align neighbors")
        ratio = 0.2
        for ref_tile, nns in neighbors.items():
            print(ref_tile)
            for nn_tile in nns:
                print(f".. {nn_tile}")

                ref_index, nn_index = ref_tile.index, nn_tile.index
                if ref_index[-1] < nn_index[-1]:
                    # x
                    ref_r, nn_r = (
                        ceil(ref_tile.data.shape[-1] * (1 - ratio)),
                        floor(nn_tile.data.shape[-1] * ratio),
                    )
                    ref_reg = ref_tile.data[:, ref_r:]
                    nn_reg = nn_tile.data[:, :nn_r]
                else:
                    # y
                    ref_r, nn_r = (
                        ceil(ref_tile.data.shape[-1] * (1 - ratio)),
                        floor(nn_tile.data.shape[-1] * ratio),
                    )
                    ref_reg = ref_tile.data[ref_r:, :]
                    nn_reg = nn_tile.data[:nn_r, :]

                ##nn_reg = match_histograms(ref_reg, nn_reg)

                shift, error, _ = register_translation(
                    ref_reg, nn_reg, upsample_factor=8, return_error=True
                )
                print(f"{ref_index} <- {nn_index}, shifts:{shift}, error:{error:04f}")

                # convert offset from region to block
                if ref_index[-1] < nn_index[-1]:
                    # x
                    offset = (0, ref_r)
                else:
                    # y
                    offset = (ref_r, 0)
                offset = [o + s for o, s in zip(offset, shift)]

                # position relative to reference tile
                pos = [p + o for p, o in zip(ref_tile.display.pos(), offset[::-1])]
                nn_tile.display.setPos(pg.Point(*pos))

                # force screen update
                pg.QtGui.QApplication.processEvents()

            print()

        """
        # calculate link score
        shifts = dict()
        for index_ref, index_tar in self._link_generator():
            # TODO currently, treat each data as 2d image
            im_ref = self._mapping[index_ref]
            im_tar = self._mapping[index_tar]

            im_tar = match_histograms(im_ref, im_tar)

            shift, error, _ = register_translation(im_ref, im_tar)
            print(f"{index_ref} <- {index_tar}, shifts:{shift}, error:{error:04f}")

            shifts[(index_ref, index_tar)] = (shift, error)

            if error < 0.3:
                print(".. apply")
                pos_ref = images[index_ref].pos()
                pos_tar = pos_ref + pg.Point(*shift)
                images[index_tar].setPos(pos_tar)

                break
            # image = pg.ImageItem(im_tar)
            # image.setPos(pg.Point(*shift))
            # plot.addItem(image)
        """

        app.instance().exec_()

    def adjust_intensity(self):
        pass

    def fuse(self):
        pass

    ##

    """
    . - . - . - . -. 4
    |   |   |   |  | 5
    . - . - . - . -. 4
    |   |   |   |  | 5
    . - . - . - . -. 4
                     22
    """

    def _list_neighbors(self):
        # index-tile lookup table
        tlut = {t.index: t for t in self.tiles}

        neighbors = dict()
        for index in product(*[range(n) for n in self.tile_shape]):
            nn = []
            for i, n in enumerate(self.tile_shape):
                nnindex = list(index)
                nnindex[i] += 1
                if nnindex[i] < n:
                    # reference <- target
                    nn.append(tlut[tuple(nnindex)])
            neighbors[tlut[index]] = nn

        return neighbors


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    ds_dir = find_dataset_dir("405")
    logger.info(f'found dataset directory "{ds_dir}"')

    files = glob.glob(os.path.join(ds_dir, "*.tif"))
    data = [imageio.imread(f) for f in files]
    logger.info(f"loaded {len(files)} tiles")

    coords = pd.read_csv(os.path.join(ds_dir, "coords.csv"), names=["x", "y", "z"])
    print(coords)

    stitcher = Stitcher(coords[:5], data[:5])
    stitcher.align()
