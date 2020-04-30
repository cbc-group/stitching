import logging
import coloredlogs
import numpy as np
import pandas as pd
from PIL import Image

from stitching.layout import Layout
from stitching.viewer import Viewer
from stitching.stitcher import Stitcher
from stitching.tiles import Tile, TileCollection

datadir = 'demo_2D_9x7_WSI'
indexfn = 'TileConfiguration.registered.txt'

logger = logging.getLogger(__name__)

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

def read_index(idxfn):
    tileheads = []
    f = open(idxfn, "rt")
    for line in f:
        line = line[:-1]
        if (line.find('.tif') < 0):
            continue
        tiffn, _, coord = line.split('; ')
        x, y = (coord[1:])[:-1].split(', ')
        tilehead = { 'file': tiffn, 'x': float(x), 'y': float(y) }
        tileheads.append(tilehead)
    f.close()
    return tileheads

def load_data(ddir, idxfn):
    tileheads = read_index(ddir + '/' + idxfn)
    x = []
    y = []
    data = []
    for tilehead in tileheads:
        im = Image.open(datadir + '/' + tilehead['file'])
        imarr = np.array(im)
        x.append(tilehead['x'])
        y.append(tilehead['y'])
        data.append(imarr)
    min = np.amin(np.asarray(x))
    if (min < 0):
        shft = np.round(-min)+1.0
        x = [ c+shft for c in x ]
    min = np.amin(np.asarray(y))
    if (min < 0):
        shft = np.round(-min)+1.0
        y = [ c+shft for c in y ]

    coords = pd.DataFrame({'x': x, 'y': y})
    return coords, data


coords, data = load_data(datadir, indexfn)

layout = Layout.from_coords(coords, maxshift=60)

# viewer = Viewer()
# viewer.show()
# collection = TileCollection(layout, data, viewer)

collection = TileCollection(layout, data, None)

stitcher = Stitcher(collection)
# stitcher.align()

chunk_shape = (1024,1024)
stitcher.fuse('output', chunk_shape)
