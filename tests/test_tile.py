from unittest import TestCase
from Tile import Tile
from Tile import Tile_state_e


class TestTile(TestCase):

    def test_init(self) -> None:
        tile = Tile(Tile_state_e.FREE)
        self.assertTrue(tile.state == Tile_state_e.FREE)

    def test_getState(self) -> None:
        tile = Tile(Tile_state_e.FREE)
        self.assertTrue(tile.state == tile.getState())
        
    def test_setBlocked(self) -> None:
        tile = Tile(Tile_state_e.FREE)
        tile.setBlocked()
        self.assertTrue(tile.state == Tile_state_e.BLOCKED)

    def test_setFree(self) -> None:
        tile = Tile(Tile_state_e.BLOCKED)
        tile.setFree()
        self.assertTrue(tile.state == Tile_state_e.FREE)
