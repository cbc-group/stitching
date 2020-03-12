import logging

import click
import coloredlogs

from utoolbox.io.dataset import BigDataViewerDataset, open_dataset

__all__ = []

logger = logging.getLogger(__name__)


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
@click.option(
    "-d",
    "--dry",
    "dry_run",
    is_flag=True,
    default=False,
    help="Dry run, generate XML only.",
)
@click.option(
    "-s",
    "--downsample",
    "downsamples",
    nargs=3,
    type=int,
    multiple=True,
    default=[(1, 1, 1), (1, 8, 8)],
    help='downsample ratio along "X Y Z" axis',
)
def main(
    src_dir,
    dst_dir,
    remap="xyz",
    flip="",
    dry_run=False,
    downsamples=[(1, 1, 1), (1, 8, 8)],
):
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

    views = src_ds.index.get_level_values("view").unique().values
    logger.info(f"{len(views)} view(s)")

    channels = src_ds.index.get_level_values("channel").unique().values
    logger.info(f"{len(channels)} channel(s)")

    # preview summary
    print(src_ds.inventory)

    # ensure downsamples is wrapped
    if isinstance(downsamples[0], int):
        downsamples = [downsamples]

    # reverse downsampling ratio for display
    _downsamples = [tuple(reversed(s)) for s in downsamples]
    logger.info(f"{len(downsamples)} pyramid level(s), {_downsamples}")

    logger.info("dumping...")
    BigDataViewerDataset.dump(
        dst_dir,
        src_ds,
        downsamples,
        chunks=(16, 64, 64),
        compression="gzip",
        dry_run=dry_run,
    )
