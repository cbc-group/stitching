import logging

import pyqtgraph as pg

__all__ = ["Viewer"]

logger = logging.getLogger(__name__)


class Viewer(object):
    def __init__(self, title="Preview"):
        # create window
        window = pg.GraphicsLayoutWidget()
        window.setWindowTitle(title)
        self._window = window

        # create viewbox
        vb = window.addViewBox()
        vb.setAspectLocked()
        vb.invertY()
        vb.enableAutoRange()

        window.show()

    ##

    @property
    def viewbox(self):
        return self._viewbox

    @property
    def window(self):
        return self._window

    ##

    def set_visible(self, flag):
        self.window.setVisible(flag)


if __name__ == "__main__":
    import glob
    import os

    import coloredlogs
    import imageio
    import pandas as pd

    from stitching.utils import find_dataset_dir

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

    viewer = Viewer()

    app = pg.mkQApp()
    app.instance().exec_()
