"""
    @file test_arena.py
    @brief unit tests for the arena class
    @author Graham Riches
    @details
   
"""
import unittest
from core.arena import Arena
from core.tile import TileState


class TestArena(unittest.TestCase):
    def setUp(self):
        # note: this means coordinates go from x = [0-4], y = [0-9]
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

    def test_set_reserved(self):
        x_values = list(range(3))
        y_values = [3]
        self.arena.set_reserved(x_values, y_values)
        for x in x_values:
            for y in y_values:
                self.assertEqual(TileState.RESERVED, self.arena.get_tile_state(x, y))

    def test_get_dimensions(self):
        dimensions = self.arena.get_dimensions()
        self.assertTupleEqual((5, 10), dimensions)

    def test_clear_blockage(self):
        x_values = list(range(3))
        y_values = [3]
        self.arena.set_blockage(x_values, y_values)
        self.arena.clear_blockage(x_values, y_values)
        for x in x_values:
            for y in y_values:
                self.assertEqual(TileState.FREE, self.arena.get_tile_state(x, y))

    def test_boundary_checking(self):
        x_range = list(range(5))
        y_range = list(range(10))
        # test the upper and lower rows
        for x in x_range:
            self.assertTrue(self.arena.is_boundary_tile(x, 0))
            self.assertTrue(self.arena.is_boundary_tile(x, 9))
        # test the sides
        for y in y_range:
            self.assertTrue(self.arena.is_boundary_tile(0, y))
            self.assertTrue(self.arena.is_boundary_tile(4, y))

    def test_neighbours_at_corners(self):
        corners = [(0, 0), (0, 9), (4, 9), (4, 0)]
        test_neighbours = [[(1, 0), (0, 1)], [(0, 8), (1, 9)], [(3, 9), (4, 8)], [(3, 0), (4, 1)]]
        for idx, corner in enumerate(corners):
            neighbours = self.arena.get_neighbours(corner[0], corner[1])
            # each corner should only have two neighbours
            self.assertEqual(2, len(neighbours))
            for check in test_neighbours[idx]:
                self.assertTrue((check in neighbours))

    def test_set_agent_target(self):
        self.arena.set_agent_target(4, 4)
        state = self.arena.get_tile_state(4, 4)
        self.assertEqual(TileState.AGENT_TARGET, state)

