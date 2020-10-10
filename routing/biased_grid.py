"""
    @file biased_grid.py
    @brief creates a directionally biased grid that can be used to alter pathing to prefer specific directions
    @author Graham Riches
    @details
        creates 2D array of integers that control preference of grid locations in routing algorithms
"""
import numpy as np
from enum import IntEnum


class BiasedDirection(IntEnum):
    BIAS_NONE = 0
    BIAS_X_POSITIVE = 1  # soft bias towards this direction
    BIAS_X_NEGATIVE = 2  # soft bias towards this direction
    BIAS_Y_POSITIVE = 3  # soft bias towards this direction
    BIAS_Y_NEGATIVE = 4  # soft bias towards this direction
    ONLY_X_POSITIVE = 5  # routing guaranteed only this direction
    ONLY_X_NEGATIVE = 6  # routing guaranteed only this direction
    ONLY_Y_POSITIVE = 7  # routing guaranteed only this direction
    ONLY_Y_NEGATIVE = 8  # routing guaranteed only this direction


class BiasedGrid:
    def __init__(self, dimensions: tuple) -> None:
        """
        Creates a biased grid object with specific 2D dimensions
        :param dimensions: tuple specifying the size of the biased grid (x, y)
        """
        # default is to have no weight preference
        self._grid = np.zeros(shape=dimensions, dtype=int)

    @property
    def grid(self) -> np.ndarray:
        """
        returns the array of weights
        :return: np.ndarray
        """
        return self._grid

    @grid.setter
    def grid(self, weights: np.ndarray) -> None:
        if np.shape(weights) == np.shape(self.grid):
            self._grid = weights

    def __setitem__(self, location: tuple, value: int) -> None:
        """
        set an item in the grid directly
        :param x_index: the x index to set
        :param y_index: the y index to set
        :return: None
        """
        x, y = location
        self._grid[x, y] = value

    def __getitem__(self, location: tuple) -> int:
        x, y = location
        return self._grid[x, y]
