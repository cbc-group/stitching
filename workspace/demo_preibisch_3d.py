import glob
import logging
import os

import dask.array as da
import imageio
import numpy as np
from dask import delayed
from dask.distributed import Client, as_completed
import xml.etree.ElementTree as ET

logger = logging.getLogger("stitcher.demo")


def launch_local_cluster():
    client = Client(
        nthreads=4,
        **{
            "memory_limit": "2GB",
            "memory_target_fraction": 0.6,
            "memory_spill_fraction": False,
            "memory_pause_fraction": 0.8,
        },
    )

    logger.info(client)

    return client


def convert_to_zarr(src_dir, dst_dir=None):
    """
    Convert their dataset to nested Zarr group dataset, since original data is already 
    renamed as C-order, we simply index them accordingly as groups.
    """
    files = glob.glob(os.path.join(src_dir, "*.tif"))
    files.sort()

    assert len(files) > 0, "no file to convert"

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

    if dst_dir is None:
        # normalize path
        parent, dname = os.path.split(src_dir)
        dname = f"{src_dir}.zarr"
        dst_dir = os.path.join(parent, dname)
        logger.debug(f'default output "{dst_dir}"')

    # open dataset, overwrite if exists
    dst_dir = zarr.open(dst_dir, mode="w")

    # input data is (x:2, y:3) tiles
    tasks = []
    i = 0
    for iy in range(3):
        for ix in range(2):
            src_arr = arrays[i]
            i += 1

            path = f"{iy:03d}/{ix:03d}"
            group = dst_dir.require_group(path)

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

    n_failed = 0
    for i in range(0, len(tasks), batch_size):
        futures = [client.compute(task) for task in tasks[i : i + batch_size]]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                n_failed += 1
                logger.exception(f"{n_failed} failed task(s)")

    return dst_dir


def load_coords_from_bdv_xml(xml_path):
    """
    Preibisch's dataset recorded in BDV XML format, all coordinates are calibrated to 
    isotropic microns. This function restore them back to voxel unit.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # TODO get voxel size

    # we only need these transform matrix
    transform_list = ["Stitching Transform", "Translation to Regular Grid"]

    for transforms in root.iter("ViewRegistration"):
        attrib = transforms.attrib
        logger.debug(f"setup {attrib['setup']}")

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
        print(matrix)

        print()


def main():
    # dataset = convert_to_zarr("/scratch/preibisch_3d/C1")
    coords = load_coords_from_bdv_xml(
        "/scratch/preibisch_3d/grid-3d-stitched-h5/dataset.xml"
    )


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    client = launch_local_cluster()
    try:
        main()
    except Exception:
        logger.exception("unexpected termination")
    finally:
        client.close()
