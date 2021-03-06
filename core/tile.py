"""
    @file tile.py
    @brief Tile object representing one unit of 2D space
    @author Evan Morcom
"""
from enum import Enum


class TileState(Enum):
    BLOCKED = 1
    FREE = 2
    RESERVED = 3
    AGENT_TARGET = 4


class Tile:
    def __init__(self, initial_state: TileState) -> None:
        """
        Create a Tile with the given state.
        :param state: The initial state of the Tile
        """
        self.state = initial_state

    def get_state(self) -> TileState:
        """
        Get the current state of a Tile
        :return state: The current state of the Tile
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

    def set_reserved(self) -> None:
        """
        set the state of a Tile to RESERVED
        """
        self.state = TileState.RESERVED

    def set_agent_target(self) -> None:
        """
        set the state of a tile to AGENT_TARGET, which means an agent has marked it
        as a destination square
        """
        self.state = TileState.AGENT_TARGET
