__all__ = []


class Layout(object):
    def __init__(self, indices=None, coords=None):
        self._indices, self._coords = indices, coords

    @classmethod
    def from_layout(cls, shape, order, unit_shape=None):
        if unit_shape is not None:
            # TODO recalculate coord in pixels
            pass

    @classmethod
    def from_coords(cls, coords):
        if all(c in coords.column for c in 'zyx'):
            pass
        # convert real scale to rank
        ranks = coords.rank(axis="index", method="dense")
        ranks = ranks.astype(int) - 1

        return cls()

    ##
    @property
    def coords(self):
        return self._coords

    @property
    def indices(self):
        return self._indices

    ##

    ##
