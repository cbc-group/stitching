import logging

from stitching.layout import Layout

__all__ = ["Tile", "TileCollection"]

logger = logging.getLogger(__name__)


class Tile(object):
    def __init__(self, index, coord, data):
        self._index, self._coord = list(index), list(coord)
        self._data = data

        # use set_viewer to ensure coord is initialized
        self._handle = None

    def __str__(self):
        index = ", ".join([f"{i:4d}" for i in self.index[::-1]])
        coord = ", ".join([f"{c:.1f}" for c in self.coord[::-1]])
        return f"<Tile ({index}) @ ({coord})"

    ##

    @property
    def coord(self):
        return tuple(self._coord)

    @property
    def data(self):
        return self._data

    @property
    def handle(self):
        return self._handle

    @property
    def index(self):
        return tuple(self._index)

    ##

    def overlap_roi(self, tile):
        pass

    def shift(self, offset):
        pass

    def set_viewer(self, viewer: "Viewer"):
        self._handle = viewer.add_image(self.data)
        logger.debug(f"add {str(self)} to viewer")
        # shift to current position
        self.handle.setPos(*self.coord[::-1][:2])

        # force update
        viewer.update()


class TileCollection(object):
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
        return self.tiles[tuple(key)]

    ##

    @property
    def layout(self):
        return self._layout

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def tiles(self):
        return self._tiles.values()

    ##

    def neighbor_of(self, tile):
        """
        Args:
            tile (list of [Tile or tuple of int]): tile to query
        
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
                # prev
                if index[i] + 1 < n:
                    nnindex = list(index)
                    nnindex[i] += 1
                    nn.append(tuple(nnindex))
                # next
                if index[i] - 1 >= 0:
                    nnindex = list(index)
                    nnindex[i] -= 1
                    nn.append(tuple(nnindex))

            # save neighbors for current index
            neighbors[index] = nn

        self._neighbors = neighbors

