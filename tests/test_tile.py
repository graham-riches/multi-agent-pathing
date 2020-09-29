from unittest import TestCase
from core.tile import Tile
from core.tile import TileState


class TestTile(TestCase):

    def test_init(self) -> None:
        tile = Tile(TileState.FREE)
        self.assertTrue(tile.state == TileState.FREE)

    def test_get_state(self) -> None:
        tile = Tile(TileState.FREE)
        self.assertTrue(tile.state == tile.get_state())
        
    def test_set_blocked(self) -> None:
        tile = Tile(TileState.FREE)
        tile.set_blocked()
        self.assertTrue(tile.state == TileState.BLOCKED)

    def test_set_free(self) -> None:
        tile = Tile(TileState.BLOCKED)
        tile.set_free()
        self.assertTrue(tile.state == TileState.FREE)

    def test_set_reserved(self):
        tile = Tile(TileState.FREE)
        tile.set_reserved()
        self.assertTrue(tile.state == TileState.RESERVED)

    def test_set_agent_target(self):
        tile = Tile(TileState.FREE)
        tile.set_agent_target()
        self.assertEqual(TileState.AGENT_TARGET, tile.state)
