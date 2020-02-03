import glob
import logging
import os

import coloredlogs
import imageio
import pandas as pd

from stitching.layout import Layout
from stitching.stitcher import Stitcher
from stitching.tiles import TileCollection
from stitching.viewer import Viewer
from stitching.utils import find_dataset_dir

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

ds_dir = find_dataset_dir("405")
logger.info(f'found dataset directory "{ds_dir}"')

files = glob.glob(os.path.join(ds_dir, "*.tif"))
files.sort()
data = [imageio.imread(f) for f in files]
logger.info(f"loaded {len(files)} tiles")

coords = pd.read_csv(os.path.join(ds_dir, "coords.csv"), names=["x", "y", "z"])
coords *= 1000 / 0.155  # px/unit
layout = Layout.from_layout((35, 40), (1, 1), (), 0.1, False)

viewer = Viewer()
viewer.show()
collection = TileCollection(layout, data, viewer)

stitcher = Stitcher(collection)
stitcher.adjust_intensity()

app = pg.mkQApp()
app.instance().exec_()
