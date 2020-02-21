import logging
import os
from concurrent.futures import ThreadPoolExecutor

import click
import coloredlogs
import dask.array as da
from dask.distributed import Client, LocalCluster
import imageio
from tqdm import tqdm

from utoolbox.cli.prompt import prompt_options
from utoolbox.io.dataset import open_dataset


logger = logging.getLogger(__name__)


def run(ds, size_limit=4096):
    # estimate resize ratio, no larger than 4k
    tile_shape, (im_shape, im_dtype) = ds.tile_shape, ds._load_array_info()
    shape = tuple(t * i for t, i in zip(tile_shape, im_shape))
    logger.debug(f"original preview {shape}, {im_dtype}")
    ratio, layer_shape = 1, shape[1:] if len(shape) == 3 else shape
    while True:
        if all((s // ratio) > size_limit for s in layer_shape):
            logger.debug(f"ratio={ratio}, exceeds size limit ({size_limit})")
            ratio *= 2
        else:
            break
    logger.info(f"target downsampling {ratio}x")

    # retrieve tiles
    def retrieve(tile):
        data = ds[tile]

        sampler = (slice(None, None, ratio),) * 2
        if data.ndim == 3:
            # data = data.max(axis=0)  # flatten
            # normally, we don't sub-sample z
            sampler = (slice(None, None, None),) + sampler
        data = data[sampler]

        return data

    def groupby_3d_tiles():
        logger.info("3D tiling dataset")
        layers = []
        for tz, tile_xy in ds.groupby(["tile_z"]):
            layer = []
            for ty, tile_x in tile_xy.groupby("tile_y"):
                row = []
                for tx, tile in tile_x.groupby("tile_x"):
                    row.append(retrieve(tile))
                layer.append(row)
            layers.append(layer)
        return layers

    def groupby_2d_tiles():
        logger.info("2D tiling dataset")
        layer = []
        for ty, tile_x in ds.groupby("tile_y"):
            row = []
            for tx, tile in tile_x.groupby("tile_x"):
                row.append(retrieve(tile))
            layer.append(row)
        return layer

    if "tile_z" in ds.index.names:
        groupby = groupby_3d_tiles
    else:
        groupby = groupby_2d_tiles
    preview = da.block(groupby())

    return preview


def load_dataset(src_dir, remap, flip):
    ds = open_dataset(src_dir)
    if len(remap) > 1 and remap != "xyz"[: len(remap)]:
        remap = {a: b for a, b in zip("xyz", remap)}
        ds.remap_tiling_axes(remap)
    if flip:
        ds.flip_tiling_axes(list(flip))
    return ds


@click.command()
@click.argument("src_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("dst_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("-r", "--remap", type=str, default="xyz")
@click.option("-f", "--flip", type=str, default="")
@click.option("-h", "--host", type=str, default="10.109.20.6:8786")
def main(src_dir, dst_dir, remap, flip, host):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    src_ds = load_dataset(src_dir, remap, flip)
    desc = tuple(
        f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(src_ds.tile_shape))
    )
    logger.info(f"tiling dimension ({', '.join(desc)})")

    try:
        views = src_ds.index.get_level_values("view").unique().values
        if len(views) > 1:
            view = prompt_options("Please select a view: ", views)
            src_ds.drop(
                src_ds.iloc[src_ds.index.get_level_values("view") != view].index,
                inplace=True,
            )
            logger.debug(f'found multiple views, using "{view}"')
        else:
            logger.debug(f"single-view dataset")
    except KeyError:
        # no need to differentiate different view
        logger.debug("not a multi-view dataset")

    try:
        channels = src_ds.index.get_level_values("channel").unique().values
        if len(channels) > 1:
            channel = prompt_options("Please select a channel: ", channels)
            src_ds.drop(
                src_ds.iloc[src_ds.index.get_level_values("channel") != channel].index,
                inplace=True,
            )
            logger.debug(f'found multiple channels, using "{channel}"')
        else:
            logger.debug(f"single-channel dataset")
    except KeyError:
        # no need to differentiate different view
        logger.debug("not a multi-channel dataset")

    # preview summary
    print(src_ds.inventory)

    # create directives
    preview = run(src_ds)
    logger.info(f"final preview {preview.shape}, {preview.dtype}")

    logger.info(f'saving preivew to "{dst_dir}"')
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        logger.warning(f'"{dst_dir}" exists')
        pass

    pbar = tqdm(enumerate(preview), total=preview.shape[0])

    def task(i, layer):
        pbar.set_description(f"layer {i+1}")
        layer = layer.compute()
        imageio.imwrite(os.path.join(dst_dir, f"layer_{i+1:04d}.tif"), layer)

    if host == "local":
        client = LocalCluster()
    else:
        client = Client(host)
    logger.info(client)

    try:
        with ThreadPoolExecutor() as pool:
            for i, layer in pbar:
                pool.submit(task, i, layer)
    except KeyboardInterrupt:
        logger.info(f"keyboard interrupted")
    finally:
        client.close()

