import logging

import click
import coloredlogs
import dask.array as da
from dask.distributed import Client, progress
import imageio
from utoolbox.cli.prompt import prompt_options
from utoolbox.io.dataset import LatticeScopeTiledDataset

__all__ = ["preview_mip", "preview_midplane"]

logger = logging.getLogger(__name__)


def preview_mip(ds):
    """Preview maximum intensity projection of each tiled-stack."""
    layers = []
    for z, tile_xy in ds.groupby("tile_z"):
        print(z)
        layer = []
        for y, tile_x in tile_xy.groupby("tile_y"):
            row = []
            for x, tile in tile_x.groupby("tile_x"):
                data = ds[tile]
                if data.ndim == 2:
                    logger.warning(f"MIP does not make sense with 2D data")
                else:
                    data = data.max(axis=0)  # mip
                row.append(data)
            layer.append(row)
        layers.append(layer)
    preview = da.block(layers)
    return preview


def preview_midplane(ds):
    """Preview midplane in the global coordinate."""
    z = ds.index.get_level_values("tile_z").unique().values
    cz = z[len(z) // 2]
    logger.info(f"x mid-plane @ {cz}")

    # select the tiles
    tiles = ds.iloc[ds.index.get_level_values("tile_z") == cz]

    layer = []
    for y, tile_x in tiles.groupby("tile_y"):
        row = []
        for x, tile in tile_x.groupby("tile_x"):
            data = ds[tile]
            if len(data.shape) > 2:
                data = data[data.shape[0] // 2, ...]  # mid plane of the stack
            row.append(data)
        layer.append(row)
    preview = da.block(layer)
    return preview


def load_dataset(src_dir, remap, flip):
    ds = LatticeScopeTiledDataset(src_dir)
    if remap != "xyz":
        remap = {a: b for a, b in zip("xyz", remap)}
        ds.remap_tiling_axes(remap)
    if flip:
        ds.flip_tiling_axes(list(flip))
    return ds


@click.command()
@click.argument("method", type=click.Choice(["mip", "midplane"], case_sensitive=False))
@click.argument("src_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("dst_path", type=click.Path(dir_okay=False))
@click.option("-r", "--remap", type=str, default="xyz")
@click.option("-f", "--flip", type=str, default="")
@click.option("-h", "--host", type=str, default="10.109.20.6:8786")
def main(method, src_dir, dst_path, remap, flip, host):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    client = Client(host)
    logger.info(client)

    src_ds = load_dataset(src_dir, remap, flip)
    desc = tuple(
        f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(src_ds.tile_shape))
    )
    logger.info(f"tiling dimension ({', '.join(desc)})")

    views = src_ds.index.get_level_values("view").unique().values
    if len(views) > 0:
        view = prompt_options("Please select a view: ", views)
        src_ds.drop(
            src_ds.iloc[src_ds.index.get_level_values("view") != view].index,
            inplace=True,
        )
        logger.debug(f'found multiple views, using "{view}"')

    print(src_ds.inventory)

    # create directives
    action = {"mip": preview_mip, "midplane": preview_midplane}[method]
    preview = action(src_ds)
    logger.info(f"preview result {preview.shape}, {preview.dtype}")

    # execute on cluster
    logger.info("generating preview...")
    preview = preview.persist()
    progress(preview)
    # retrieve
    preview = preview.compute()

    logger.info(f'saving preivew to "{dst_path}"')
    try:
        imageio.volwrite(dst_path, preview)
    except ValueError:
        imageio.imwrite(dst_path, preview)

    client.close()

