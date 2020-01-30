import logging

import numpy as np
import pandas as pd

__all__ = []

logger = logging.getLogger(__name__)


class Layout(object):
    def __init__(self, indices, coords):
        self._indices, self._coords = indices, coords
        self._tile_shape = tuple(len(set(s)) for s in zip(*self.indices))
        logger.info(f"{self.tile_shape} tiles")

    @classmethod
    def from_layout(cls, tile_shape, direction, data_shape, overlap, snake=False):
        """
        Args:
            tile_shape (tuple of int): number of tiles in each dimension
            direction (tuple of int): tiling direction, describe by +/-1
            data_shape (tuple of int): shape of a single tile
            overlap (float or tuple of float): overlap ratio in each dimension
            snake (bool, optional): walk through axes consecutively
        """
        if snake:
            # requires direction toggling
            direction = list(direction)

        overlap = tuple(overlap)
        if len(overlap) == 1:
            overlap *= len(tile_shape)  # expand to fit ndims
        # shrink shape by overlap ratio
        data_shape = tuple(s * (1 - o) for s, o in zip(data_shape, overlap))

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

        def walk(index, axis):
            while True:
                yield tuple(index)
                index, overflow = step(index, axis)
                if overflow:
                    # current axis overflow
                    for _axis in range(axis - 1, -1, -1):
                        index, overflow = step(index, _axis)
                        if not overflow:
                            break
                    else:
                        return

        # index cursor
        i_cursor = [t - 1 if d < 0 else 0 for t, d in zip(tile_shape, direction)]
        logger.debug(f"cursor init index {tuple(i_cursor)}")
        # create index maps
        indices = [c for c in walk(i_cursor, len(tile_shape) - 1)]

        # multiply tile shapes to get pixel coordinates
        coords = [tuple(ii * s for ii, s in zip(i, data_shape)) for i in indices]

        # create instance
        return cls(indices, coords)

    @classmethod
    def from_coords(cls, coords):
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

        # convert real scale to rank
        ranks = coords.rank(axis="index", method="dense")
        ranks = ranks.astype(int) - 1

        # pack as list of tuples
        indices = [tuple(r.values.astype(np.uint16)) for _, r in ranks.iterrows()]
        coords = [tuple(c.values.astype(np.float32)) for _, c in coords.iterrows()]

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
