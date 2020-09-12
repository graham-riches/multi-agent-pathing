"""
    @file test_render_engine
    @brief tests for the rendering engine. Currently quite bad
    @author Graham Riches
    @details
   
"""
import unittest
from arena import Arena
from render_engine import Canvas


TIMESTEP = 0.01


class TestArenaRenderer(unittest.TestCase):
    def setUp(self):
        self.arena = Arena(5, 10)
        self.canvas = Canvas(self.arena, TIMESTEP)

    def test_render(self):
        # Note: this test requires manually closing the window. Yuck
        self.arena.set_blockage([1], [1])
        self.arena.set_blockage([2], [2])
        self.canvas.run(test_mode=True)
