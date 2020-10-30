import os

import imageio
import zarr

zarr_dir = "output.zarr"
tiff_dir = "output.tiff"

try:
    os.mkdir(tiff_dir)
except FileExistsError:
    pass

dataset = zarr.open(zarr_dir, "r")
print(f"shape={dataset.shape}, dtype={dataset.dtype}")

if len(dataset.shape) == 2:
    path = os.path.join(tiff_dir, f"layer_00000.tif")
    imageio.imwrite(path, dataset)
else:
    for i, layer in enumerate(dataset):
        print(f"layer: {i}")
        path = os.path.join(tiff_dir, f"layer_{i+1:05d}.tif")
        imageio.imwrite(path, layer)
