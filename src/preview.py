import logging
import os

import imageio
import numpy as np
from tqdm import tqdm
import zarr

__all__ = ["preview_layers"]

logger = logging.getLogger(__name__)


def preview_layers(zarr_path):
    array = zarr.open(zarr_path, mode="r")
    logger.debug(f"dataset shape: {array.shape}")

    # output location
    parent, basename = os.path.split(zarr_path)
    basename = os.path.splitext(basename)[0]
    out_dir = os.path.join(parent, f"{basename}_layers")
    try:
        os.makedirs(out_dir)
    except FileNotFoundError:
        pass

    # iterate over slowest dimension (Z)
    nz = array.shape[0]
    for iz in tqdm(range(nz), total=nz):
        layer = array[iz, ...]
        path = os.path.join(out_dir, f"layer_{iz:03d}.tif")
        imageio.imwrite(path, layer)


if __name__ == "__main__":
    import coloredlogs

    from utils import find_dataset_dir

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    root = find_dataset_dir("trial_4")
    zarr_path = os.path.join(root, "trial_4.zarr")
    script_path = os.path.join(root, "volume.csv")

    preview_layers(zarr_path)
