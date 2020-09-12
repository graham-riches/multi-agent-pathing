"""
    @file tile.py
    @brief Tile object representing one unit of 2D space
    @author Evan Morcom
"""
from enum import Enum


class TileState(Enum):
    BLOCKED = 1
    FREE = 2


class Tile:
    def __init__(self, initial_state: TileState) -> None:
        """
        Create a Tile with the given state.
        :param state: The initial state of the Tile (blocked or free)
        """
        self.state = initial_state

    def get_state(self) -> TileState:
        """
        Get the current state of a Tile
        :return state: The current state of the Tile (blocked or free)
        """   
        return self.state
    
    def set_blocked(self) -> None:
        """
        Set the state of a Tile to blocked
        """
        self.state = TileState.BLOCKED

    def set_free(self) -> None:
        """
        Set the state of a Tile to free
        """
        self.state = TileState.FREE
