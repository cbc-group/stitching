import logging

from dask.distributed import Client

from utoolbox.io.dataset import open_dataset, BigDataViewerDataset

__all__ = []

logger = logging.getLogger(__name__)


def main(src_dir, dst_dir):
    ds = open_dataset(src_dir)
    ds.remap_tiling_axes({"x": "y", "y": "x"})  # 6F, W.C. setup

    print(ds.inventory)

    logger.info("dumping...")
    BigDataViewerDataset.dump(
        dst_dir, ds, [(1, 4, 4)], chunks=(16, 64, 64), compression="gzip"
    )


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # client = Client("10.109.20.6:8786")
    # logger.info(client)

    main(
        "X:/vins/ExM_kidney/20200130_ExM_kidney_7_d_k5_Gelatin_FITC_lectin488_11x16_z50um_1",
        "V:/Vins/20200130_ExM_kidney_7_d_k5_Gelatin_FITC_11x16_z50um_1_bin4",
    )
