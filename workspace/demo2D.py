import dask.array as da
from dask.distributed import Client, as_completed

import logging
import coloredlogs
import numpy as np
import pandas as pd
import zarr
import imageio
# from PIL import Image
from prompt_toolkit.shortcuts import progress_dialog

from stitching.layout import Layout
from stitching.viewer import Viewer
from stitching.stitcher import Stitcher
from stitching.tiles import Tile, TileCollection

data_dir  = '../data/demo_2D_9x7_WSI'
data_zdir = '../data/demo_2D_9x7_WSI.zarr'
out_zdir  = 'output.zarr'
indexfn   = 'TileConfiguration.registered.txt'
logger    = logging.getLogger(__name__)


def launch_cluster(address=None):
    if address is None:
        client = Client(
            nthreads=4,
            **{
                "memory_limit": "2GB",
                "memory_target_fraction": 0.6,
                "memory_spill_fraction": False,
                "memory_pause_fraction": 0.8,
            },
        )
    else:
        client = Client(address)
    logger.info(client)
    return client


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
        imarr = imageio.imread(data_dir + '/' + tilehead['file'])
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


def convert_to_zarr(dst_dir, data, layout, overwrite=False):
    mode = "w" if overwrite else "r"
    writeback = overwrite
    try:
        dataset = zarr.open(dst_dir, mode=mode)
    except ValueError as err:
        if mode == "r":
            writeback = True
            dataset = zarr.open(dst_dir, mode="w")
        else:
            logger.exception(f'{dst_dir}: {err}')
            quit()

    if not writeback:
        return dataset

    indices = layout.indices
    tasks = []
    for index, src_arr in zip(indices, data):
        src_arr = da.from_array(src_arr, chunks=(256, 256))
        desc = [ f"{i:03d}" for i in index ]
        path = '/'.join(desc)
        group = dataset.require_group(path)

        # create container
        dst_arr = group.require_dataset("raw", src_arr.shape, chunks=(256,256), dtype=src_arr.dtype, exact=True)

        # dump data
        src_arr = src_arr.rechunk()
        task = src_arr.to_zarr(
               dst_arr, overwrite=True, compute=False, return_stored=False)
        tasks.append(task)

    # use active workers to determine batch size
    client = Client.current()
    batch_size = len(client.ncores())

    def worker(set_percentage, log_text):
        n_completed, n_failed = 0, 0
        for i in range(0, len(tasks), batch_size):
            futures = [client.compute(task) for task in tasks[i:i+batch_size]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    n_failed += 1
                    logger.exception(f"{n_failed} failed task(s)")
                else:
                    n_completed += 1
                    set_percentage((n_completed + n_failed) / len(tasks))

    # start write / overwrite the zarr source.
    progress_dialog(text="Convert to Zarr", run_callback=worker).run()

    return dataset


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(level="DEBUG",
        fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    client = launch_cluster(address="localhost:8786")

    coords, data = load_data(data_dir, indexfn)
    layout = Layout.from_coords(coords, maxshift=60)
    viewer = None
    dataset = convert_to_zarr(data_zdir, data, layout, overwrite=False)

    data = []
    for _, yx in dataset.groups():
        for _, x in yx.groups():
            array = da.from_zarr(x["raw"])
            if False:
                # DEBUG force load data in memory
                array = np.array(array)
            data.append(array)
    collection = TileCollection(layout, data, viewer)

    stitcher = Stitcher(collection)
    # stitcher.align()

    stitcher.fuse(out_zdir, overwrite=True)
