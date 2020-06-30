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
    files = glob.glob(os.path.join(src_dir, "*.tif"))
    files.sort()

    assert len(files) > 0, "no file to convert"

    if dst_dir is None:
        # normalize path
        src_dir = os.path.abspath(src_dir)
        parent, dname = os.path.split(src_dir)
        dname = f"{src_dir}.zarr"
        dst_dir = os.path.join(parent, dname)
        logger.debug(f'default output "{dst_dir}"')

    # open dataset, overwrite if exists
    mode = "w" if overwrite else "w-"
    try:
        dataset = zarr.open(dst_dir, mode=mode)
    except ValueError as err:
        if "contains a" in str(err):
            if overwrite is None:
                print(dst_dir)
                raise FileExistsError(dst_dir)
            else:
                # return as read-only
                logger.info(f"load existing dataset")
                return zarr.open(dst_dir, mode="r")

    # open arbitrary item to determine array info
    tmp = imageio.volread(files[0])
    shape, dtype = tmp.shape, tmp.dtype
    logger.info(f"tile shape {shape}, {dtype}")
    del tmp

    def volread_np(uri):
        """Prevent serialization error."""
        return np.array(imageio.volread(uri))

    def volread_da(uri):
        """Create dask array from delayed image reader."""
        return da.from_delayed(delayed(volread_np, pure=False)(uri), shape, dtype)

    arrays = [volread_da(uri) for uri in files]

    # input data is (x:2, y:3) tiles
    tasks = []
    i = 0
    for iy in range(3):
        for ix in range(2):
            src_arr = arrays[i]
            i += 1

            path = f"{iy:03d}/{ix:03d}"
            group = dataset.require_group(path)

            # create container
            dst_arr = group.require_dataset("raw", shape, dtype=dtype, exact=True)

            # dump data
            src_arr = src_arr.rechunk()
            task = src_arr.to_zarr(
                dst_arr, overwrite=True, compute=False, return_stored=False
            )
            tasks.append(task)

    # use active workers to determine batch size
    client = Client.current()
    batch_size = len(client.ncores())

    def worker(set_precentage, log_text):
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
                    set_precentage((n_completed + n_failed) / len(tasks))

    progress_dialog(text="Convert to Zarr", run_callback=worker).run()

    return dataset


def load_coords_from_bdv_xml(xml_path):
    """
    Preibisch's dataset recorded in BDV XML format, all coordinates are calibrated to 
    isotropic microns. This function restore them back to voxel unit.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    setup = next(root.iter("ViewSetup"))
    voxel_size = setup.find("voxelSize").find("size").text
    voxel_size = [float(value) for value in voxel_size.split(" ")]
    voxel_size = np.array(voxel_size, dtype=np.float32)

    # since transform matrix is isotropic, we divide it by the smallest voxel dimension
    voxel_size = voxel_size[0]
    logger.info(f"voxel size {voxel_size} um")

    # we only need these transform matrix
    transform_list = ["Stitching Transform", "Translation to Regular Grid"]

    coords = []
    for transforms in root.iter("ViewRegistration"):
        matrix = np.identity(4, dtype=np.float32)
        for transform in transforms.iter("ViewTransform"):
            # determine types
            name = transform.find("Name").text

            if name not in transform_list:
                continue

            # extract array
            values = transform.find("affine").text
            values = [float(value) for value in values.split(" ")] + [0, 0, 0, 1]
            values = np.array(values, dtype=np.float32).reshape((4, 4))

            # accumulate
            matrix = np.matmul(matrix, values)

        # shift vector is the last column (F-order)
        shift = matrix[:-1, -1]

        # TODO z dimension is upsampled, divide it?

        # save it as list in order to batch convert later
        coords.append(shift)
    # there are 3 channels, we use the first one
    coords = coords[:6]

    # as pandas dataframe
    coords = pd.DataFrame(coords, columns=["x", "y", "z"], dtype=np.float32)
    print(coords)

    return coords


def main():
    # load zarr dataset
    src_dir = "data/preibisch_3d/C1"
    try:
        # dataset = convert_to_zarr(src_dir)
        # DEBUG force read-only
        dataset = convert_to_zarr(src_dir, overwrite=False)
    except FileExistsError as err:
        overwrite = yes_no_dialog(
            title="Zarr dataset exists",
            text=f"Should we overwrite the existing dataset?\n({err})",
        ).run()
        dataset = convert_to_zarr(src_dir, overwrite=overwrite)

    # load coordinates
    coords = load_coords_from_bdv_xml(
        "data/preibisch_3d/grid-3d-stitched-h5/dataset.xml"
    )

    # repopulate the dataset as list of dask array
    data = []
    for _, yx in dataset.groups():
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
    stitcher.fuse(dst_dir, (64, 64, 64), overwrite=True)  # TODO why chunk shape


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    client = launch_cluster(address=None)
    try:
        main()
    except Exception:
        logger.exception("unexpected termination")
    finally:
        client.close()
