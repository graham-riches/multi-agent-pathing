"""
    @file Tile.py
    @brief Tile object representing one unit of 2D space
    @author Evan Morcom
"""
import numpy as np
from enum import Enum

class Tile_state_e(Enum):
    BLOCKED = 1
    FREE = 2

class Tile:
    def __init__(self, initial_state: Tile_state_e) -> None:
        """
        Create a Tile with the given state.
        :param state: The initial state of the Tile (blocked or free)
        """
        self.state = initial_state

    def getState(self) -> Tile_state_e:
        """
        Get the current state of a Tile
        :return state: The current state of the Tile (blocked or free)
        """   
        return self.state
    
    def setBlocked(self) -> None:
        """
        Set the state of a Tile to blocked
        """
        self.state = Tile_state_e.BLOCKED

    def setFree(self) -> None:
        """
        Set the state of a Tile to free
        """
        self.state = Tile_state_e.FREE
        