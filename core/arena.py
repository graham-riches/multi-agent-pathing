"""
    @file arena.py
    @brief contains the main 2D routing arena
    @author Graham Riches
    @details
    creates a 2D Arena of tiles that can be used for path finding etc.
   
"""
from core.tile import Tile, TileState
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
                self._grid[int(x)][int(y)] = Tile(TileState.FREE)

    def set_blockage(self, x_values: list, y_values: list) -> None:
        """
        set a routing blockage over a range of coordinates
        :param x_values: list of x coordinates
        :param y_values: list of y coordinates
        :return: None
        """
        for x in x_values:
            for y in y_values:
                self._grid[int(x)][int(y)].set_blocked()

    def clear_blockage(self, x_values: list, y_values: list) -> None:
        """
        clear a routing blockage
        :param x_values: x coordinates to clear
        :param y_values: y coordinates to clear
        :return: None
        """
        for x in x_values:
            for y in y_values:
                self._grid[int(x)][int(y)].set_free()

    def set_reserved(self, x_values: list, y_values: list) -> None:
        """
        reserve tiles for routing
        :param x_values: x coordinates to reserve
        :param y_values: y coordinates to reserve
        :return: None
        """
        for x in x_values:
            for y in y_values:
                self._grid[int(x)][int(y)].set_reserved()

    def set_agent_target(self, x_location: int, y_location: int) -> None:
        """
        Reserve a square as an agents target square
        :param x_location: x location
        :param y_location: y location
        :return:
        """
        self._grid[(int(x_location))][int(y_location)].set_agent_target()

    def get_tile_state(self, x_location: int, y_location: int) -> TileState:
        """
        Get the tile state for a specific location
        :param x_location: the x location
        :param y_location: the y location
        :return: tile state
        """
        return self._grid[int(x_location)][int(y_location)].get_state()

    def is_boundary_tile(self, x_location: int, y_location: int) -> bool:
        """
        check if a tile is at the edge of the arena
        :param x_location: x coordinate
        :param y_location: y coordinate
        :return: boolean True/False
        """
        x_is_edge = self.is_edge_x(x_location)
        y_is_edge = self.is_edge_y(y_location)
        return x_is_edge or y_is_edge

    def is_edge_x(self, x_location: int) -> bool:
        """
        test if an x coordinate is an edge
        :param x_location:
        :return:
        """
        if (x_location < (self._x_size - 1)) and (x_location > 0):
            x_is_edge = False
        else:
            x_is_edge = True
        return x_is_edge

    def is_edge_y(self, y_location: int) -> bool:
        """
        test if an y coordinate is an edge
        :param y_location:
        :return:
        """
        if (y_location < (self._y_size - 1)) and (y_location > 0):
            y_is_edge = False
        else:
            y_is_edge = True
        return y_is_edge

    def get_neighbours(self, x_location: int, y_location: int) -> list:
        """
        Get all of a tiles valid neighbours
        :param x_location: x-coordinate
        :param y_location: y_coordinate
        :return: list of neighbour coordinate tuples
        """
        x_is_edge = self.is_edge_x(x_location)
        y_is_edge = self.is_edge_y(y_location)
        neighbours = list()
        # handle all edges and corner cases (heh)
        if x_is_edge and y_is_edge:
            # add the two closest nodes
            if x_location == 0:
                node_1 = (int(x_location + 1), int(y_location))
                node_2 = (int(x_location), int(y_location + 1)) if y_location == 0 else (int(x_location), int(y_location - 1))
            else:
                node_1 = (int(x_location - 1), int(y_location))
                node_2 = (int(x_location), int(y_location + 1)) if y_location == 0 else (int(x_location), int(y_location - 1))
            neighbours.append(node_1)
            neighbours.append(node_2)
        elif x_is_edge and not y_is_edge:
            # add the upper and lower neighbours
            neighbours.append((int(x_location), int(y_location + 1)))
            neighbours.append((int(x_location), int(y_location - 1)))
            # add the internal neighbour
            if x_location == 0:
                neighbours.append((int(x_location + 1), int(y_location)))
            else:
                neighbours.append((int(x_location - 1), int(y_location)))
        elif y_is_edge and not x_is_edge:
            # add the upper and lower neighbours
            neighbours.append((int(x_location + 1), int(y_location)))
            neighbours.append((int(x_location - 1), int(y_location)))
            # add the internal neighbour
            if y_location == 0:
                neighbours.append((int(x_location), int(y_location + 1)))
            else:
                neighbours.append((int(x_location), int(y_location - 1)))
        else:
            # add all surrounding nodes
            neighbours.append((int(x_location - 1), int(y_location)))
            neighbours.append((int(x_location + 1), int(y_location)))
            neighbours.append((int(x_location), int(y_location - 1)))
            neighbours.append((int(x_location), int(y_location + 1)))
        return neighbours
