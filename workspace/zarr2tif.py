import os

import imageio
import zarr

zarr_dir = "../data/preibisch_3d/C1_output.zarr"
tiff_dir = "../data/preibisch_3d/C1_output.tiff"

try:
    os.mkdir(tiff_dir)
except FileExistsError:
    pass

dataset = zarr.open(zarr_dir, "r")
print(f"shape={dataset.shape}, dtype={dataset.dtype}")

for i, layer in enumerate(dataset):
    path = os.path.join(tiff_dir, f"layer_{i+1:05d}.tif")
    imageio.imwrite(path, layer)
