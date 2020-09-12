"""
    @file test_arena.py
    @brief unit tests for the arena class
    @author Graham Riches
    @details
   
"""
import unittest
from arena import Arena
from tile import TileState


class TestArena(unittest.TestCase):
    def setUp(self):
        self.arena = Arena(5, 10)

    def test_generate_grid(self):
        self.assertEqual(TileState.FREE, self.arena.get_tile_state(4, 9))

    def test_set_blockage(self):
        x_values = list(range(3))
        y_values = [3]
        self.arena.set_blockage(x_values, y_values)
        for x in x_values:
            for y in y_values:
                self.assertEqual(TileState.BLOCKED, self.arena.get_tile_state(x, y))

    def test_get_dimensions(self):
        dimensions = self.arena.get_dimensions()
        self.assertTupleEqual((5, 10), dimensions)
