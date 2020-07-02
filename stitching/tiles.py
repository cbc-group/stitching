import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import linregress
from dask import delayed
from stitching.layout import Layout

__all__ = ["Tile", "TileCollection"]

logger = logging.getLogger(__name__)


class Tile:
    def __init__(self, index, coord, data):
        self._index, self._coord = list(index), np.array(coord, dtype=np.float32)
        self._data = data

        # init intensity profile
        self.brightness, self.contrast = 0.0, 1.0

        # use set_viewer to ensure coord is initialized
        self._handle, self._viewer = None, None

    def __str__(self):
        index = ", ".join([f"{i:d}" for i in self.index[::-1]])
        coord = ", ".join([f"{c:.1f}" for c in self.coord[::-1]])
        return f"<Tile ({index}) @ ({coord})>"

    ##

    @property
    def brightness(self):
        return np.float32(self._brightness)

    @brightness.setter
    def brightness(self, value):
        self._brightness = np.float32(value)
        # TODO trigger updates

    @property
    def contrast(self):
        return np.float32(self._contrast)

    @contrast.setter
    def contrast(self, value: float):
        self._contrast = np.float32(value)
        # TODO trigger updates

    @property
    def coord(self) -> np.ndarray:
        return self._coord

    @property
    def data(self):
        return self.contrast * self._data + self.brightness

    @property
    def handle(self):
        return self._handle

    @property
    def index(self) -> Tuple[int]:
        return tuple(self._index)

    ##

    def overlap_roi(self, tile: "Tile", return_raw_roi=False):
        """
        Returns region that overlap with a provided tile.

        Args:
            tile (Tile): tile to compare with
            return_raw_roi (bool, optional): return ROI coordinate in global coordinate
        
        Returns:
            (Tile or None): returns a view if overlapped, otherwise, None
        """
        #   up left
        #   (x0, y0) ----- +
        #      |           |
        #      + ------ (x1, y1)
        #               down right

        # up-left
        a_coord0, b_coord0 = np.array(self.coord), np.array(tile.coord)

        # down-right
        a_shape, b_shape = np.array(self.data.shape), np.array(tile.data.shape)
        if len(a_coord0) < len(a_shape):
            # pad missing dimensions
            a_coord0, b_coord0 = (
                np.concatenate([1], a_coord0),
                np.concatenate([1], b_coord0),
            )
        elif len(a_coord0) > len(a_shape):
            # drop excessive dimension
            a_coord0, b_coord0 = a_coord0[1:], b_coord0[1:]

        # should only differ by 1-D
        assert a_coord0.ndim == a_shape.ndim

        a_coord1, b_coord1 = a_coord0 + a_shape, b_coord0 + b_shape

        # max(up-left)
        c_coord0 = np.maximum(a_coord0, b_coord0)
        # min(down-right)
        c_coord1 = np.minimum(a_coord1, b_coord1)

        if np.any(c_coord1 <= c_coord0):
            if return_raw_roi:
                return None, None
            else:
                return None

        if return_raw_roi:
            roi = tuple(c_coord0) + tuple(c_coord1)

        # offset the roi coordinate to local coordinate
        c_coord0 -= a_coord0
        c_coord1 -= a_coord0
        logger.debug(f"ROI coordinate, top-left {c_coord0}, bottom-right {c_coord1}")
        assert np.all(c_coord0 >= 0) and np.all(
            c_coord1 <= a_shape
        ), "unknown ROI calculation error"

        # build slice
        slices = tuple(
            slice(i, j)
            for i, j in zip(c_coord0.round().astype(int), c_coord1.round().astype(int))
        )
        subregion = self.data[slices]

        if return_raw_roi:
            return subregion, roi
        else:
            return subregion

    def shift(self, offset):
        self._coord = [c + o for c, o in zip(self.coord, offset)]
        if self.handle is not None:
            self.handle.setPos(*self.coord[::-1][:2])
            # force update
            self._viewer.update()

    ##

    def reset_intensity(self):
        """
        Reset brightness offset to 0 and contrast slope to 1.
        """
        self._brightness, self._contrast = np.float32(0), np.float32(1)
        # TODO trigger updates

    def match_intensity(self, tile: "Tile", threshold=0.5, apply=True):
        """
        Match intensity of a given tile _to current tile_ by linear fitting intensity
        of overlapped region.

        Y = m X + c
            Y: overlap region of this tile
            X: overlap region of provided tile

        Args:
            tile (Tile): tile to compare with
            threshold (float, optional): if not None, raise error when correlation
                falls below threshold
            apply (bool, optional): if True, apply new contrast adjustment to the tile

        Returns:
            (tuple of float) as (m, c)
        """
        reference = self.overlap_roi(tile)
        if reference is None:
            raise ValueError(f"no overlap between {self.index} and {tile.index}")
        target = tile.overlap_roi(self)

        m, c, r2, _, _ = linregress(target.ravel(), reference.ravel())
        if r2 < threshold:
            raise ValueError(
                f"low intensity correlation between {reference.index} and {target.index} (R^2={r2:.4f})"
            )

        if apply:
            # original:
            #   y0 = m0 x + c0
            #
            # new (m, c):
            #   y = m y0 + c
            #     = m (m0 x + c0) + c
            #     = m m0 x + (m c0 + c)
            tile.contrast *= m
            tile.brightness = m * tile.brightness + c

        return m, c

    ##

    def set_viewer(self, viewer: "Viewer"):
        self._handle = viewer.add_image(self.data)
        logger.debug(f"add {str(self)} to viewer")
        # shift to current position
        self.handle.setPos(*self.coord[::-1][:2])

        # force update
        self._viewer = viewer
        viewer.update()


class TileCollection:
    def __init__(self, layout: Layout, data, viewer: "Viewer" = None):
        self._layout = layout
        self._tiles = {
            i: Tile(i, *args) for i, *args in zip(layout.indices, layout.coords, data)
        }

        # build neighbor lut
        self._build_neighbor_lut()

        # populate tiles in the viewer
        if viewer:
            for tile in self.tiles:
                tile.set_viewer(viewer)

    def __getitem__(self, key):
        return self._tiles[tuple(key)]

    ##

    @property
    def origin(self) -> Tuple[int]:
        # build list of current coordinates
        coords = [tile.coord for tile in self.tiles]
        coords = pd.DataFrame(coords, dtype=np.float32)

        return tuple(np.floor(coords.min().values).astype(int))

    @property
    def bounding_box(self) -> Tuple[int]:
        # build list of current coordinates
        coords = [tile.coord for tile in self.tiles]
        coords = pd.DataFrame(coords, dtype=np.float32)

        # build list of shapes of tiles (we cannot assume each tile has the same shape)
        shapes = [tile.data.shape for tile in self.tiles]
        shapes = pd.DataFrame(shapes, dtype=np.float32)

        # use pandas to determine boundary (anchors + tiles)
        #
        #   1           1: coords
        #   +---+
        #   |   |
        #   +---+
        #       2       2: coords + shapes
        bbox_max = (coords + shapes).max().values
        bbox_min = coords.min().values

        # round up to ensure we consider every voxels
        bbox = np.ceil(bbox_max) - np.floor(bbox_min)

        # cast to proper type
        return tuple(bbox.astype(int))

    @property
    def layout(self):
        return self._layout

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def tiles(self) -> List[Tile]:
        return list(self._tiles.values())

    ##

    def neighbor_of(self, tile):
        """
        Args:
            tile (list of [Tile or tuple of int]): tile to query
            nn (string): 'all': all neighbors; 'next': neighbor next to me.
        
        Returns:
            (list of [Tile or tuple of int])
        """
        if isinstance(tile, Tile):
            index = tile.index
        logger.debug(f"reference index {index}")
        nnindex = self.neighbors[index]
        if isinstance(tile, Tile):
            # de-reference
            return [self[i] for i in nnindex]
        else:
            return nnindex

    ##

    def _build_neighbor_lut(self):
        tile_shape = self.layout.tile_shape

        neighbors = dict()
        for index in self.layout.indices:
            nn = []
            for i, n in enumerate(tile_shape):
                # next
                if index[i] + 1 < n:
                    nnindex = list(index)
                    nnindex[i] += 1
                    nn.append(tuple(nnindex))
                # prev
                if index[i] - 1 >= 0:
                    nnindex = list(index)
                    nnindex[i] -= 1
                    nn.append(tuple(nnindex))

            # save neighbors for current index
            neighbors[index] = nn

        self._neighbors = neighbors
