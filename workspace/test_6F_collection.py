import logging

import coloredlogs
from dask.distributed import Client, progress
import imageio
import numpy as np
import pandas as pd
import pyqtgraph as pg
from utoolbox.io.dataset import LatticeScopeTiledDataset

from stitching.layout import Layout
from stitching.stitcher import Stitcher
from stitching.tiles import Tile, TileCollection
from stitching.viewer import Viewer

logger = logging.getLogger(__name__)

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

## start dask client
client = Client(processes=False)  # Client("10.109.20.6:8786")
logger.info(client)

## load lls ds
src_dir = "S:/ARod/20200121_1_2"
ds = LatticeScopeTiledDataset(src_dir)

ds.remap_tiling_axes({"x": "y", "y": "x"})

desc = tuple(f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(ds.tile_shape)))
logger.info(f"tiling dimension ({', '.join(desc)})")

## drop CamA
ds.drop(ds.iloc[ds.index.get_level_values("view") != "CamB"].index, inplace=True)
print(ds.inventory)

## retrieve first layer
z = ds.index.get_level_values("tile_z").unique().values
tiles = ds.iloc[ds.index.get_level_values("tile_z") == z[0]]

coords = []
data = []
for y, tile_x in tiles.groupby("tile_y"):
    for x, tile in tile_x.groupby("tile_x"):
        coords.append((x, y))
        data.append(ds[tile].compute())

## create tile collection
x, y = tuple(list(zip(*coords)))
dz, dy, dx = ds._load_voxel_size()
coords = pd.DataFrame({"x": x, "y": y})
coords["x"] /= dx
coords["y"] /= dy

layout = Layout.from_coords(coords)
viewer = Viewer()
viewer.show()

collection = TileCollection(layout, data, viewer)

stitcher = Stitcher(collection)
stitcher.adjust_intensity()
stitcher.align()

## event loop
app = pg.mkQApp()
app.instance().exec_()

## cleanup
client.close()
