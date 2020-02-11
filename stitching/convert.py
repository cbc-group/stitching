import logging

from utoolbox.io.dataset import open_dataset, BigDataViewerDataset

__all__ = []

logger = logging.getLogger(__name__)


def main(src_dir, dst_dir):
    ds = open_dataset(src_dir)
    ds.remap_tiling_axes({"x": "y", "y": "x"})  # 6F, W.C. setup

    logger.info("dumping...")
    BigDataViewerDataset.dump(
        dst_dir, ds, [(1, 1, 1), (1, 4, 4)], chunks=(16, 64, 64), compression="gzip"
    )


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main(
        "D:/XueiJiao/ExM_kidney/20200130_ExM_kidney_7_d_k5_Gelatin_FITC_11x16_z50um_1",
        "V:/Vins/20200130_ExM_kidney_7_d_k5_Gelatin_FITC_11x16_z50um_1",
    )
