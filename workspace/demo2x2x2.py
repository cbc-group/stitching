import logging

import coloredlogs
# from dask.distributed import Client, progress
import numpy as np
import pandas as pd
from utoolbox.io.dataset import LatticeScopeTiledDataset

from stitching.layout import Layout
from stitching.stitcher import Stitcher
from stitching.tiles import Tile, TileCollection

logger = logging.getLogger(__name__)

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

## start dask client
# client = Client(processes=False)  # Client("10.109.20.6:8786")
# logger.info(client)

## load lls ds
src_dir = "data/demo_3D_2x2x2_CMTKG-V3"
ds = LatticeScopeTiledDataset(src_dir)

ds.remap_tiling_axes({"x": "y", "y": "z", "z": "x"})
ds.flip_tiling_axes(["x", "y"])

desc = tuple(f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(ds.tile_shape)))
logger.info(f"tiling dimension ({', '.join(desc)})")

print(ds.inventory)

## retrieve first layer
# z = ds.index.get_level_values("tile_z").unique().values
# tiles = ds.iloc[ds.index.get_level_values("tile_z") == z[0]]

coords = []
data = []
for z, tile_xy in ds.groupby("tile_z"):
    for y, tile_x in tile_xy.groupby("tile_y"):
        for x, tile in tile_x.groupby("tile_x"):
            coords.append((x, y, z))
            data.append(ds[tile].compute())
print(coords)

## create tile collection
x, y, z = tuple(list(zip(*coords)))
dz, dy, dx = ds._load_voxel_size()
print(f"dz={dz}, dy={dy}, dx={dx}")

coords = pd.DataFrame({"x": x, "y": y, "z": z})
coords["x"] /= dx
coords["y"] /= dy
coords["z"] /= dz
coords["x"] -= coords["x"].min()
coords["y"] -= coords["y"].min()
coords["z"] -= coords["z"].min()
print(coords)

layout = Layout.from_coords(coords)
# raise RuntimeError('DEBUG')
collection = TileCollection(layout, data)

stitcher = Stitcher(collection)
# stitcher.adjust_intensity()
# stitcher.align()

chunk_shape = (64,64,64)
stitcher.fuse('output', chunk_shape)

## cleanup
# client.close()
