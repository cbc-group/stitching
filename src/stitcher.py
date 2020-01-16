import glob
from itertools import product
import logging
from math import ceil, floor
import os

import imageio
import numpy as np
import pandas as pd
import pyqtgraph as pg
from skimage.exposure import match_histograms
from skimage.feature import register_translation

from utoolbox.stitching.phasecorr import PhaseCorrelation

from utils import find_dataset_dir

__all__ = []

logger = logging.getLogger(__name__)


class Stitcher(object):
    def __init__(self, data, coords):
        self._mapping = self._map_data_from_coords(data, coords)

        self._tile_shape = coords.nunique().values[::-1]
        desc = ", ".join([f"{a}:{n}" for a, n in zip("xyz", self.tile_shape[::-1])])
        logger.info(f"tile shape ({desc})")

    ##

    @property
    def mapping(self):
        return self._mapping

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
        images = dict()
        template = None
        for index, data in self.mapping.items():
            # TODO assuming it is 2d
            coord = [c * n for c, n in zip(index[1:][::-1], data.shape)]

            image = pg.ImageItem(data)
            image.setPos(pg.Point(*coord))
            image.setOpts(axisOrder="row-major")
            vb.addItem(image)

            images[index] = image

            pg.QtGui.QApplication.processEvents()

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

        app.instance().exec_()

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

    def _link_generator(self):
        shape = self.tile_shape
        for coord in product(*[range(n) for n in shape]):
            for i, n in enumerate(shape):
                _coord = list(coord)
                _coord[i] += 1
                if _coord[i] < n:
                    # reference <- target
                    yield (coord, tuple(_coord))

    def _map_data_from_coords(self, data, coords):
        # convert real scale to rank
        coords = coords.rank(axis="index", method="dense")
        coords = coords.astype(int) - 1

        # associate data with relative index
        mapping = dict()
        for (_, row), d in zip(coords.iterrows(), data):
            mapping[tuple(row.values[::-1])] = d

        return mapping


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

    stitcher = Stitcher(data, coords)
    stitcher.align()
