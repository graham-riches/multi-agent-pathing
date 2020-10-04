"""
    @file test_benchmark.py
    @brief unit tests for the benchmark runner/loader classes
    @author Graham Riches
    @details
    Test the benchmark runner functionality
   
"""
import unittest

from core.benchmark import BenchmarkRunner
from core.tile import TileState


class TestBenchmarkLoader(unittest.TestCase):
    def setUp(self):
        benchmark_file = '../benchmarks/test_config.json'
        self.loader = BenchmarkRunner(benchmark_file)

    def tearDown(self):
        pass

    def test_parse_sim_properties(self):
        self.loader.parse_simulation_properties()
        self.assertEqual(0.005, self.loader.time_step)
        self.assertEqual(40, self.loader.dpi)
        self.assertFalse(self.loader.render)

    def test_parse_agents_without_sim_config_fails(self):
        self.loader.parse_agents()
        self.assertEqual(None, self.loader.agents)

    def test_parse_agents_list(self):
        self.loader.parse_simulation_properties()
        self.loader.parse_agents()
        self.assertEqual(10, len(self.loader.agents))

    def test_parse_arena(self):
        self.loader.parse_arena()
        dimensions = self.loader.arena.get_dimensions()
        self.assertTupleEqual((10, 10), dimensions)
        state = self.loader.arena.get_tile_state(3, 3)
        self.assertEqual(TileState.BLOCKED, state)

    def test_parse_tasks(self):
        self.loader.parse_tasks()
        self.assertEqual(10, len(self.loader.tasks))

    def test_no_algorithms_returns_zero(self):
        cycles, distance = self.loader.run()
        self.assertEqual(0, cycles)

