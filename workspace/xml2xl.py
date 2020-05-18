import logging
import os
import re
import xml.etree.ElementTree as et
from collections import defaultdict
from glob import glob

import coloredlogs
import numpy as np
import pandas as pd

from utoolbox.io.dataset import LatticeScopeTiledDataset as _LatticeScopeTiledDataset

logger = logging.getLogger(__name__)

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


class LatticeScopeTiledDataset(_LatticeScopeTiledDataset):
    """Override read_func to return filename only."""

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            return uri

        return func


def fix_dtype(df):
    # assign proper dtype
    dtype_table = {k: np.float32 for k in df.columns}
    # .. except filename
    dtype_table["Filename"] = str

    return df.astype(dtype_table)


def load_tile_info(root):
    # pull out attributes
    info = []
    for node in root:
        info.append(dict(node.attrib))

    # convert to DataFrame
    records = defaultdict(list)
    for node in root:
        for key, value in node.attrib.items():
            records[key].append(value)
    df = pd.DataFrame.from_dict(records)

    df = fix_dtype(df)

    return df


def main(src_dir, project_path, imaris_dir):
    xtree = et.parse(project_path)
    xroot = xtree.getroot()
    # /Extends/Image
    xroot = xroot.findall("Extends")[0]

    df = load_tile_info(xroot)
    logger.info(f"stitcher project uses {len(df)} tile(s)")

    # calculate actual size
    for ax in ("X", "Y", "Z"):
        size = df[f"Max{ax}"] - df[f"Min{ax}"]
        size = size.round().astype(int)  # we know size in pixel is discrete
        df[ax] = size

    # calculate tile index
    index = defaultdict(list)
    pattern = r"_(\d+)_x(\d{3})_y(\d{3})"
    for tile_name in df["Filename"]:
        matches = re.search(pattern, tile_name)
        assert matches is not None, "unable to interpret filename"

        for ax, value in zip(("iZ", "iX", "iY"), matches.groups()):
            value = int(value)
            index[ax].append(value)
    # write back
    for key, value in index.items():
        df[key] = pd.Series(value)

    print(">> BEFORE")
    print(df)
    print()

    # convert to full resolution coordinate
    for ax in ("X", "Y"):
        df[f"Max{ax}"] = df[f"Max{ax}"] * 2048 / df[ax]  # FIXME hard-coded 2048?
        df[f"Min{ax}"] = df[f"Min{ax}"] * 2048 / df[ax]

    # load dataset
    ds = LatticeScopeTiledDataset.load(src_dir)
    print(ds.inventory)

    # fix axes
    ds.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
    ds.flip_tiling_axes(["x", "y"])

    # create filename mapping
    for i, row in df.iterrows():
        index = tuple(row[f"i{ax}"] for ax in ("X", "Y", "Z"))

        # retrieve filename from tile index
        x, y, z = index
        index_mapping = {"tile_x": x, "tile_y": y, "tile_z": z}
        filename = ds[index_mapping]
        logger.debug(f'{index} -> "{os.path.basename(filename)}"')

        # replace
        filename = os.path.basename(filename)
        filename, _ = os.path.splitext(filename)
        filename = os.path.join(imaris_dir, filename)
        new_filename = f"{filename}.ims"

        old_filename = df.at[i, "Filename"]

        df.at[i, "OldFilename"] = old_filename
        df.at[i, "Filename"] = new_filename

    print(">> AFTER")
    with pd.option_context("display.max_columns", None):
        print(df)
    print()

    # update attributes
    for node in xroot:
        filename = node.get("Filename")

        # lookup updated attributes
        attributes = df[df["OldFilename"] == filename]
        assert len(attributes) == 1, "duplicated filename"
        new_attrs = attributes.iloc[0].to_dict()

        # replace
        for key in node.attrib.keys():
            value = str(new_attrs[key])
            node.set(key, value)

    # generate result
    fname, fext = os.path.splitext(project_path)
    dst_xml = f"{fname}_modify{fext}"
    xtree.write(dst_xml)


if __name__ == "__main__":
    main(
        src_dir="Y:/ARod/4F/20200324_No5_CamA",
        project_path="0.xml",
        imaris_dir="Y:/ARod/4F/20200324_No5_CamA/Full_resolution/layer_0",
    )
