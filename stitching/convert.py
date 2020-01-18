import glob
import logging
import os

import imageio
from numcodecs import Blosc
import numpy as np
from tqdm import tqdm
import zarr

from reader import filename_to_tile, read_script, read_settings

__all__ = []

logger = logging.getLogger(__name__)


def main(data_dir, script_path):
    tile_shape, tile_pos = read_script(script_path)
    data_shape = read_settings(data_dir)

    # estimate overall shape
    full_shape = tuple(t * a for t, a in zip(tile_shape, data_shape))
    # NOTE arbitrary for now
    chunk_shape = (1, 512, 512)

    # specify compression spec
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    out_path = f"{data_dir}.zarr"
    # create new array
    # NOTE we know camera can only provide uint16
    array = zarr.open(
        out_path,
        mode="w",
        shape=full_shape,
        chunks=chunk_shape,
        dtype="u2",
        compressor=compressor,
    )

    # loop over files and start loading
    tile_index_lut = filename_to_tile(data_dir, script_path)
    file_list = glob.glob(os.path.join(data_dir, "*.tif"))
    progress = tqdm(file_list)
    for path in progress:
        progress.set_description(f'loading "{os.path.basename(path)}"')

        index = tile_index_lut[path]
        data = imageio.imread(path)

        # convert to global position
        head = tuple(i * a for i, a in zip(index, data_shape))
        tail = tuple((i + 1) * a for i, a in zip(index, data_shape))
        print(f'head: {head}, tail: {tail}')
        # zip to slice
        selection = tuple(slice(h, t) for h, t in zip(head, tail))
        array[selection] = data[np.newaxis, ...]

    return array


if __name__ == "__main__":
    import coloredlogs

    from utils import find_dataset_dir

    logging.getLogger("tifffile").setLevel(logging.ERROR)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    root = find_dataset_dir("trial_7")
    data_dir = root
    script_path = os.path.join(root, "script.csv")

    main(data_dir, script_path)
