"""
    @file test_render_engine
    @brief tests for the rendering engine. Currently quite bad
    @author Graham Riches
    @details
   
"""
import unittest
import numpy as np
from arena import Arena
from agent import Agent
from render_engine import Renderer


TIMESTEP = 0.01
BASE_DPI_SCALING = 40


class TestRenderer(unittest.TestCase):

    def test_generate_grid(self):
        self.arena = Arena(15, 20)
        self.renderer = Renderer(self.arena, TIMESTEP, BASE_DPI_SCALING)
        grid = self.renderer.generate_grid()
        self.assertTupleEqual((20*BASE_DPI_SCALING, 15*BASE_DPI_SCALING), np.shape(grid))

    @unittest.skip('not required right now for testing')
    def test_render_arena(self):
        self.arena = Arena(15, 20)
        self.renderer = Renderer(self.arena, TIMESTEP)
        self.arena.set_blockage([1], [1])
        self.renderer.run()

    def test_rendering_with_agents(self):
        self.arena = Arena(15, 20)
        self.renderer = Renderer(self.arena, TIMESTEP)
        self.arena.set_blockage([1], [1])
        agent = Agent(4, 4, TIMESTEP)
        self.renderer.add_agent(agent)
        self.renderer.run()
        pass



