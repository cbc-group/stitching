import logging
import os
from typing import List

import click
import coloredlogs
import dask.array as da
import imageio
from dask.distributed import Client, LocalCluster, as_completed, progress
from tqdm import tqdm
from utoolbox.cli.prompt import prompt_options
from utoolbox.io import open_dataset

import zarr

logger = logging.getLogger(__name__)


def run(ds, size_limit=4096, mip=False):
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
            if mip:
                # flatten the entire tile
                data = data.max(axis=0)
            else:
                # normally, we don't sub-sample z
                sampler = (slice(None, None, None),) + sampler
        data = data[sampler]

        return data

    def groupby_tiles(inventory, index: List[str]):
        """
        Aggregation function that generates the proper internal list layout for all the tiles in their natural N-D layout.

        Args:
            inventory (pd.DataFrame): the listing inventory
            index (list of str): the column header
        """
        tiles = []
        for _, tile in inventory.groupby(index[0]):
            if len(index) > 1:
                # we are not at the fastest dimension yet, decrease 1 level
                tiles.append(groupby_tiles(tile, index[1:]))
            else:
                # fastest dimension, call retrieval function
                tiles.append(retrieve(tile))
        return tiles

    index = ["tile_y", "tile_x"]
    if "tile_z" in ds.index.names:
        index = ["tile_z"] + index
    logger.info(f"a {len(index)}-D tiled dataset")

    # pack as a huge array
    preview = da.block(groupby_tiles(ds, index))

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
@click.option("-m", "--mip", default=False, is_flag=True)
def main(src_dir, dst_dir, remap, flip, host, mip):
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

    if host == "local":
        client = LocalCluster()
    else:
        client = Client(host)
    logger.info(client)

    # create directives
    preview = run(src_ds, mip=mip)
    logger.info(f"final preview {preview.shape}, {preview.dtype}")

    # saving the result to zarr format
    zarr_path = f"{dst_dir.rstrip(os.sep)}.zarr"
    chunks = (8, 512, 512)
    logger.info(f'generating "{os.path.basename(zarr_path)}"')
    logger.debug(f"shape={preview.shape}, dtype={preview.dtype}, chunks={chunks}")

    try:
        logger.info("dumping to zarr directory store, waiting...")
        preview = preview.rechunk(chunks)
        da.to_zarr(preview, zarr_path, overwrite=False)
    except ValueError:
        logger.warning("found existing zarr store, reusing it")

    logger.info("release dask array")
    del preview

    logger.info(f'saving layered preivew to "{dst_dir}"')
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        logger.warning(f'"{dst_dir}" exists')
        pass

    logger.info(f"reload data from zarr")
    preview = da.from_zarr(zarr_path)

    futures = []
    with tqdm(total=preview.shape[0]) as pbar:
        for i, layer in enumerate(preview):
            fname = f"layer_{i+1:04d}.tif"
            pbar.set_description(fname)
            path = os.path.join(dst_dir, fname)
            future = client.submit(imageio.imwrite, path, layer)
            futures.append(future)
            pbar.update(1)

    # submit tasks
    with tqdm(
        total=len(futures), bar_format="{l_bar}{bar:24}{r_bar}{bar:-10b}"
    ) as pbar:
        for future in as_completed(futures, with_results=False):
            try:
                future.result()  # ensure we do not have an exception
                pbar.update(1)
            except Exception as error:
                logger.exception(error)
            future.release()

    logger.info("closing scheduler connection")
    client.close()
