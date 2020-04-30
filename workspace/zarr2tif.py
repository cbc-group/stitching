import numpy as np
import zarr
import imageio

zarr_dir = 'output'
tiff_fn = 'output.tif'

z = zarr.open(zarr_dir, 'r')
print(z.shape)
print(z.dtype)

imageio.imwrite(tiff_fn, z)
