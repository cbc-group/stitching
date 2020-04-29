import logging

import numpy as np
import pandas as pd

__all__ = []

logger = logging.getLogger(__name__)


class Layout(object):
    def __init__(self, indices, coords):
        self._indices, self._coords = indices, coords
        self._tile_shape = tuple(len(set(s)) for s in zip(*self.indices))

        label = "xyz" if len(self.tile_shape) == 3 else "xy"
        desc = ", ".join([f"{l}:{v}" for l, v in zip(label, self.tile_shape[::-1])])
        logger.info(f"({desc}) tiles")

    @classmethod
    def from_layout(cls, tile_shape, axis_order, direction, data_shape, overlap, snake=False):
        """
        Args:
            tile_shape (tuple of int): number of tiles in each dimension
            axis_order (string): string of any ordering of 'x', 'y', 'z'
            direction (tuple of int): tiling direction, describe by +/-1
            data_shape (tuple of int): shape of a single tile
            overlap (float or tuple of float): overlap ratio in each dimension
            snake (bool, optional): walk through axes consecutively
        """
        if snake:
            # requires direction toggling
            direction = list(direction)

        if isinstance(overlap, float):
            overlap  = (overlap,)
        if len(overlap) == 1:
            overlap *= len(tile_shape)  # expand to fit ndims
        # shrink shape by overlap ratio
        data_shape = tuple(s * (1 - o) for s, o in zip(data_shape, overlap))

        # axis_order: remove 'z' axis for 2D, and translate to axisid
        if (len(tile_shape) <= 2):
            axis_order = axis_order.replace("z", "")
            axisid = { 'x':1, 'y':0 }
        else:
            axisid = { 'x':2, 'y':1, 'z':0 }
        axis_order = [ axisid[lab] for lab in list(axis_order) ]

        # convert tile_shape from (Nx,Ny,Nz) to (Nz,Ny,Nx)
        tile_shape = tile_shape[::-1]

        def step(index, axis):
            overflow = True
            if direction[axis] < 0 and index[axis] <= 0:
                # negative overflow
                if snake:
                    # toggle direction
                    index[axis] = 0
                    direction[axis] *= -1
                else:
                    index[axis] = tile_shape[axis] - 1
            elif direction[axis] > 0 and index[axis] >= tile_shape[axis] - 1:
                # positive overflow
                if snake:
                    # toggle direction
                    index[axis] = tile_shape[axis] - 1
                    direction[axis] *= -1
                else:
                    index[axis] = 0
            else:
                # .. next step in current axis
                index[axis] += direction[axis]
                overflow = False
            return index, overflow

        def walk(index, axis_order, aidx):
            while True:
                yield tuple(index)
                index, overflow = step(index, axis_order[aidx])
                if overflow:
                    # current axis overflow
                    for _aidx in range(1, len(axis_order)):
                        index, overflow = step(index, axis_order[_aidx])
                        if not overflow:
                            break
                    else:
                        return

        # index cursor
        i_cursor = [t-1 if d < 0 else 0 for t, d in zip(tile_shape, direction)]
        logger.debug(f"cursor init index {tuple(i_cursor)}")
        # create index maps
        aidx    = 0
        indices = [c for c in walk(i_cursor, axis_order, aidx)]

        # multiply tile shapes to get pixel coordinates
        coords = [tuple(ii*s for ii, s in zip(i, data_shape)) for i in indices]

        # create instance
        return cls(indices, coords)

    @classmethod
    def from_coords(cls, coords, maxshift=10):
        """
        Args:
            coords (pd.DataFrame): coordinates in pixels
        """
        assert isinstance(
            coords, pd.DataFrame
        ), "coordinates need to store in DataFrame"

        if any(c not in coords.columns for c in "xy"):
            raise ValueError('must have "x" and "y" column')
        headers = list("zyx" if "z" in coords.columns else "yx")

        # reorder columns
        coords = coords[headers]

        # pack as list of tuples
        coords = [tuple(c.values.astype(np.float32)) for _, c in coords.iterrows()]

        # fix indices to be the tile ranks
        def _coord_to_indices(_coords, _maxshift):
            ndim = len(_coords[0])
            idxs = []
            for i in range(ndim):
                cs = sorted([coord[i] for coord in _coords])
                c0 = cs[0]
                ds = []
                i0 = 0
                for cc in cs:
                    i0 += (cc-c0 > _maxshift)
                    ds.append(i0)
                    c0 = cc
                idx = [ ds[cs.index(coord[i])] for coord in _coords ]
                for i,id in enumerate(idx):
                    if (i > 0 and id-idx[i-1] > 1):
                        idx[i] = idx[i-1]+1
                idxs.append(tuple(idx))
            return [i for i in zip(*idxs)]

        indices = _coord_to_indices(coords, maxshift)

        # create instance
        return cls(indices, coords)

    ##
    @property
    def coords(self):
        return self._coords

    @property
    def indices(self):
        return self._indices

    @property
    def tile_shape(self):
        return self._tile_shape

    ##

    ##
