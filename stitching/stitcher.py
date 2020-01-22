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
from scipy.optimize import minimize
from skimage.exposure import histogram, match_histograms, equalize_adapthist
from skimage.feature import register_translation

from stitching.tiles import Tile
from stitching.utils import find_dataset_dir

__all__ = []

logger = logging.getLogger(__name__)


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

        # montage
        logger.info("generate montage")
        t = []  # TEST
        for tile in self.tiles:
            print(tile.index)

            # create display
            image = pg.ImageItem(tile.data)
            image.setOpts(axisOrder="row-major")
            vb.addItem(image)
            tile.display = image

            t.append(tile.data)  # TEST

            # shift to default position
            # TODO assuming it is 2d
            coord = [c * n for c, n in zip(tile.index[1:][::-1], tile.data.shape)]
            image.setPos(pg.Point(*coord))

            # force screen update
            pg.QtGui.QApplication.processEvents()

        t = np.array(t)  # TEST

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
                    ref_reg, nn_reg, upsample_factor=1, return_error=True
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

                """
                # extract overlapped region
                #
                #   (x0, y0) ----- +
                #      |           |
                #      + ------ (x1, y1)
                #
                #   (y0, x0, y1, x1)
                ref_roi = (0, 0) + ref_tile.data.shape
                pos = [int(p) for p in pos[::-1]]
                nn_roi = pos + [p + s for p, s in zip(pos, nn_tile.data.shape)]
                roi = (
                    max(ref_roi[0], nn_roi[0]),
                    max(ref_roi[1], nn_roi[1]),
                    min(ref_roi[2], nn_roi[2]),
                    min(ref_roi[3], nn_roi[3]),
                )
                ny, nx = roi[2] - roi[0], roi[3] - roi[1]
                logger.debug(f"overlap roi {roi}, shape:{(ny, nx)}")
                # crop
                logger.debug(f"ref roi {roi}")
                ref_reg = ref_tile.data[roi[0] : roi[2], roi[1] : roi[3]]
                roi = [
                    r - o for r, o in zip(roi, (roi[0], roi[1]) * 2)
                ]  # offset back to nn coordinate
                logger.debug(f"nn roi {roi}")
                nn_reg = nn_tile.data[roi[0] : roi[2], roi[1] : roi[3]]
                logger.debug(f"{ref_reg.shape}, {nn_reg.shape}")

                imageio.imwrite("ref.tif", ref_reg)
                imageio.imwrite("nn.tif", nn_reg)

                # map neighbor intensity to reference by linear transformation
                func = lambda x: np.sum(((x[0] * nn_reg + x[1]) - ref_reg) ** 2)
                res = minimize(
                    func, [1, 0], method="Nelder-Mead", options={"disp": True}
                )
                m, c = tuple(res.x)
                print(res)

                # ref_reg, nn_reg = ref_reg.ravel(), nn_reg.ravel()
                # A = np.vstack([nn_reg, np.ones(len(nn_reg))]).T
                # m, c = np.linalg.lstsq(A, ref_reg, rcond=None)[0]
                logger.debug(f".. y={m:.2f}x+{c:.2f}")

                nn_reg = m * nn_reg + c
                nn_reg = nn_reg.reshape(ny, nx)
                imageio.imwrite("nn_corr.tif", nn_reg.astype(np.float32))

                # recalculate and apply
                data = nn_tile.data
                ##data = m * data + c
                """
                # data = match_histograms(nn_tile.data, t.reshape(2048 * 3, 2048 * 5))
                data = equalize_adapthist(nn_tile.data)
                nn_tile.display.setImage(data)

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
    files.sort()
    data = [imageio.imread(f) for f in files]
    logger.info(f"loaded {len(files)} tiles")

    coords = pd.read_csv(os.path.join(ds_dir, "coords.csv"), names=["x", "y", "z"])
    print(coords)

    nlim = 2
    stitcher = Stitcher(coords[:nlim], data[:nlim])
    stitcher.align()
