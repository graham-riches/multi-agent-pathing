"""
    @file arena.py
    @brief contains the main 2D routing arena
    @author Graham Riches
    @details
    creates a 2D Arena of tiles that can be used for path finding etc.
   
"""
from tile import Tile, TileState
import numpy as np


class Arena:
    def __init__(self, x_size: int, y_size: int) -> None:
        """
        Creates a new arena with dimensions given by the tuple (x,y). For now this is just a 2D
        rectangular grid, but this can be modified to a more general grid type
        :param: x_size: number of x tiles
        :param: y_size: number of y_tiles
        """
        self._grid = None
        self._x_size = x_size
        self._y_size = y_size
        self.generate_grid(x_size, y_size)

    def get_dimensions(self) -> tuple:
        """
        get the dimensions of the arena as a tuple
        :return:
        """
        return self._x_size, self._y_size

    def generate_grid(self, x_units: int, y_units: int) -> None:
        self._grid = np.ndarray((x_units, y_units), dtype=object)
        for x in range(x_units):
            for y in range(y_units):
                self._grid[x][y] = Tile(TileState.FREE)

    def set_blockage(self, x_values: list, y_values: list) -> None:
        """
        set a routing blockage over a range of coordinates
        :param x_values: list of x coordinates
        :param y_values: list of y coordinates
        :return: None
        """
        for x in x_values:
            for y in y_values:
                self._grid[x][y].set_blocked()

    def get_tile_state(self, x_location: int, y_location: int) -> TileState:
        """
        Get the tile state for a specific location
        :param x_location: the x location
        :param y_location: the y location
        :return: tile state
        """
        return self._grid[x_location][y_location].get_state()



