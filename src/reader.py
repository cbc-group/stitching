import glob
import logging
import os
import re

import pandas as pd

__all__ = ["filename_to_tile", "read_script", "read_settings"]

logger = logging.getLogger(__name__)


def read_script(script_path):
    # summary section
    df = pd.read_csv(script_path, nrows=1)
    tile_shape = tuple(df[f"# of Subvolume Z Stacks {ax}"][0] for ax in ("Z", "Y", "X"))
    logger.info(f"tile shape: {tile_shape}")

    # actual position
    df = pd.read_csv(script_path, skiprows=2)
    # sort by tile spatial index, fastest dimension first
    df = df.sort_values(by=[f"Stack {ax}" for ax in ("X", "Y", "Z")])

    return (
        tile_shape,
        {
            tuple(index): tuple(pos)
            for index, pos in zip(
                df[[f"Stack {ax}" for ax in ("Z", "Y", "X")]].values,
                df[[f"Absolute {ax} (um)" for ax in ("Z", "Y", "X")]].values,
            )
        },
    )


def filename_to_tile(data_dir, script_path):
    """
    Map filename to tile index.

    Args:
        data_dir (str): path to the raw data folder
        script_path (str): path to the script that generated scanning steps

    Returns:
        (tuple of int): (Z, Y, X)
    """
    file_list = glob.glob(os.path.join(data_dir, "*.tif"))
    file_list.sort()

    # actual position
    df = pd.read_csv(script_path, skiprows=2)
    # ... only keep index
    df = df[[f"Stack {ax}" for ax in ("Z", "Y", "X")]]

    return {fname: tuple(row) for fname, row in zip(file_list, df.values)}


def read_settings(data_dir):
    """
    Read SPIM generated setting file.

    Returns:
        (tuple of int): (Z, Y, X)
    """
    file_list = glob.glob(os.path.join(data_dir, "*_Settings.txt"))
    if len(file_list) > 1:
        logger.warning("found multiple setting file, ignored")
    setting_path = file_list[0]

    # parse image size
    with open(setting_path, "r") as fd:
        for line in fd:
            matches = re.match(r"# of Pixels :\s+X=(\d+) Y=(\d+)", line)
            if matches is None:
                continue
            # we know z will only have 1 layer
            return 1, matches.group(2), matches.group(1)


if __name__ == "__main__":
    import coloredlogs

    from utils import find_dataset_dir

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    path = find_dataset_dir("trial_4")
    path = os.path.join(path, "volume.csv")
    print(path)
    print(read_script(path))
