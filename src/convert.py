import glob
import logging
import os

import pandas as pd
import zarr

from reader import read_script, read_settings

__all__ = []

logger = logging.getLogger(__name__)


def main(data_dir, script_path):
    tile_shape, tile_pos = read_script(script_path)
    data_shape = read_settings(data_dir)
    out_path = f"{data_dir}.zarr"
    # create new array


if __name__ == "__main__":
    import coloredlogs

    from utils import find_dataset_dir

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    root = find_dataset_dir("trial_4")
    data_dir = os.path.join(root, "trial_4")
    script_path = os.path.join(root, "volume.csv")

    main(data_dir, script_path)
