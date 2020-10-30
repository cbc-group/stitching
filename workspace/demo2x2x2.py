import glob
import logging
import os
import xml.etree.ElementTree as ET

import dask.array as da
import imageio
import numpy as np
import pandas as pd
import zarr
from dask import delayed
from dask.distributed import Client, as_completed
from prompt_toolkit.shortcuts import progress_dialog, yes_no_dialog
from utoolbox.io.dataset import TiledDatasetIterator
from utoolbox.io import open_dataset

from stitching.layout import Layout
from stitching.stitcher import Stitcher
from stitching.tiles import Tile, TileCollection

logger = logging.getLogger("stitcher.demo")


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


def convert_to_zarr(src_dir, dst_dir=None, overwrite=None):
    """
    Convert their dataset to nested Zarr group dataset, since original data is already 
    renamed as C-order, we simply index them accordingly as groups.
    """
    ds = open_dataset(src_dir)

    if dst_dir is None:
        # normalize path
        src_dir = os.path.abspath(src_dir)
        parent, dname = os.path.split(src_dir)
        dname = f"{src_dir}.zarr"
        dst_dir = os.path.join(parent, dname)
        logger.debug(f'default output "{dst_dir}"')

    # open dataset, overwrite if exists
    mode = "w" if overwrite else "w-"
    writeback = True
    try:
        dataset = zarr.open(dst_dir, mode=mode)
    except ValueError as err:
        if "contains a" in str(err):
            if overwrite is None:
                raise FileExistsError(dst_dir)
            else:
                writeback = False
                # return as read-only
                logger.info(f"load existing dataset")
                dataset = zarr.open(dst_dir, mode="r")

    coords = []
    tasks = []
    for (index, coord), tile in TiledDatasetIterator(
        ds, axes="zyx", return_key=True, return_format="both"
    ):
        coords.append(coord)
        if not writeback:
            continue
        src_arr = ds[tile]

        desc = [f"{i:03d}" for i in index]
        path = "/".join(desc)
        group = dataset.require_group(path)

        # create container
        dst_arr = group.require_dataset(
            "raw", src_arr.shape, dtype=src_arr.dtype, exact=True
        )

        # dump data
        src_arr = src_arr.rechunk()
        task = src_arr.to_zarr(
            dst_arr, overwrite=True, compute=False, return_stored=False
        )
        tasks.append(task)

    coords = pd.DataFrame(coords, columns=["z", "y", "x"], dtype=np.float32)
    print(coords)

    # Data ready for existing zarr source and no overwrite.
    if len(tasks) == 0:
        return coords,dataset

    # use active workers to determine batch size
    client = Client.current()
    batch_size = len(client.ncores())

    def worker(set_percentage, log_text):
        n_completed, n_failed = 0, 0
        for i in range(0, len(tasks), batch_size):
            futures = [client.compute(task) for task in tasks[i : i + batch_size]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    n_failed += 1
                    logger.exception(f"{n_failed} failed task(s)")
                else:
                    n_completed += 1
                    set_percentage((n_completed + n_failed) / len(tasks))

    progress_dialog(text="Convert to Zarr", run_callback=worker).run()

    return coords,dataset


# def load_coords_from_bdv_xml(xml_path):
#     """
#     Preibisch's dataset recorded in BDV XML format, all coordinates are calibrated to 
#     isotropic microns. This function restore them back to voxel unit.
#     """
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
# 
#     setup = next(root.iter("ViewSetup"))
#     voxel_size = setup.find("voxelSize").find("size").text
#     voxel_size = [float(value) for value in voxel_size.split(" ")]
#     voxel_size = np.array(voxel_size, dtype=np.float32)
# 
#     # since transform matrix is isotropic, we divide it by the smallest voxel dimension
#     voxel_size = voxel_size[0]
#     logger.info(f"voxel size {voxel_size} um")
# 
#     # we only need these transform matrix
#     transform_list = ["Stitching Transform", "Translation to Regular Grid"]
# 
#     coords = []
#     for transforms in root.iter("ViewRegistration"):
#         matrix = np.identity(4, dtype=np.float32)
#         for transform in transforms.iter("ViewTransform"):
#             # determine types
#             name = transform.find("Name").text
# 
#             if name not in transform_list:
#                 continue
# 
#             # extract array
#             values = transform.find("affine").text
#             values = [float(value) for value in values.split(" ")] + [0, 0, 0, 1]
#             values = np.array(values, dtype=np.float32).reshape((4, 4))
# 
#             # accumulate
#             matrix = np.matmul(matrix, values)
# 
#         # shift vector is the last column (F-order)
#         shift = matrix[:-1, -1]
# 
#         # TODO z dimension is upsampled, divide it?
# 
#         # save it as list in order to batch convert later
#         coords.append(shift)
#     # there are 3 channels, we use the first one
#     coords = coords[:6]
# 
#     # as pandas dataframe
#     coords = pd.DataFrame(coords, columns=["x", "y", "z"], dtype=np.float32)
#     print(coords)
# 
#     return coords


def main():
    # load zarr dataset
    src_dir = "/home2/rescue/stitch/run/data/demo_3D_2x2x2_CMTKG-V3"
    try:
        # dataset = convert_to_zarr(src_dir)
        # DEBUG force read-only
        coords, dataset = convert_to_zarr(src_dir, overwrite=False)
    except FileExistsError as err:
        overwrite = yes_no_dialog(
            title="Zarr dataset exists",
            text=f"Should we overwrite the existing dataset?\n({err})",
        ).run()
        coords, dataset = convert_to_zarr(src_dir, overwrite=overwrite)

    # load coordinates
#    coords = load_coords_from_bdv_xml(
#        "data/preibisch_3d/grid-3d-stitched-h5/dataset.xml"
#    )

    # repopulate the dataset as list of dask array
    data = []
    for _, zyx in dataset.groups():
        for _, yx in zyx.groups():
            for _, x in yx.groups():
                array = da.from_zarr(x["raw"])
                if False:
                    # DEBUG force load data in memory
                    array = np.array(array)
                data.append(array)

    # create stitcher
    layout = Layout.from_coords(coords)
    collection = TileCollection(layout, data)
    stitcher = Stitcher(collection)

    # rebuild output dir
    # normalize path
    src_dir = os.path.abspath(src_dir)
    parent, dname = os.path.split(src_dir)
    dname = f"{src_dir}_output.zarr"
    dst_dir = os.path.join(parent, dname)

    # execute
    stitcher.fuse(dst_dir)


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    client = launch_cluster(address="localhost:8786")
    try:
        main()
    except Exception:
        logger.exception("unexpected termination")
    finally:
        client.close()
