from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["Tile"]


@dataclass
class Tile(object):
    index: Tuple[int]
    coord: Tuple[float]
    data: np.ndarray

    ##

    def __hash__(self):
        return hash(self.index)

    def __str__(self):
        desc = ", ".join([f"{k}:{v:.3f}" for k, v in zip("xyz", self.coord[::-1])])
        return f"<Tile {self.index} @ ({desc})"


class TileCollection(object):
    def __init__(self):
        pass