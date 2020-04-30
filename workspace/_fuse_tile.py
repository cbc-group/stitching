import numpy as np
from scipy.interpolate import RegularGridInterpolator

def func(xx, yy):
    return np.sin((xx*np.pi/3)**2+(yy*np.pi/4)**2)

coord0 = (1.04,1.13)
shape0 = (3,4)
x = np.linspace(coord0[0],coord0[0]+shape0[0]-1,shape0[0])
y = np.linspace(coord0[1],coord0[1]+shape0[1]-1,shape0[1])
print(x)
print(y)
xx, yy = np.meshgrid(x, y, indexing="ij")
data0 = np.sin(xx**2+yy**2)
print(data0)

vol_shape = (6,6)
vol = np.zeros(vol_shape)

class mytile:
    def __init__(self, _coord, _data):
        print(_data.shape)
        self.coord = _coord
        self.data  = _data

tile = mytile(coord0, data0)


def _fuse_tile(vol, tile):
    data = tile.data
    dshape = data.shape
    coord0 = tile.coord
    axes_mesh0 = tuple(np.linspace(x,x+L-1,L) for x, L in zip(coord0, dshape))
    fusefunc = RegularGridInterpolator(axes_mesh0, data, bounds_error=False, fill_value=None)

    coord1 = tuple(np.round(x).astype(int) for x in coord0)
    mesh1 = np.meshgrid(*tuple(np.linspace(x,x+L-1,L) for x, L in zip(coord1, dshape)), indexing='ij')
    pts1 = fusefunc([ pt for pt in zip(*(x.flat for x in mesh1)) ])
    pts1[pts1 < 0] = 0
    pxls1 = np.round(pts1).astype(np.uint16).reshape(dshape)
    if (len(dshape) == 2):
        vol[coord1[0]:coord1[0]+dshape[0],
            coord1[1]:coord1[1]+dshape[1]] = pxls1
    else:
        vol[coord1[0]:coord1[0]+dshape[0],
            coord1[1]:coord1[1]+dshape[1],
            coord1[2]:coord1[2]+dshape[2]] = pxls1

_fuse_tile(vol, tile)
print(vol)
