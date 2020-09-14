"""
    @file test_a_star.py
    @brief unit tests for A* path finding
    @author Graham Riches
    @details
   
"""
import unittest
from arena import Arena
from agent import Agent
from routing.a_star import AStar, Node
from routing.status import RoutingStatus


class TestAStar(unittest.TestCase):
    def setUp(self):
        time_step = 0.005
        self.arena = Arena(10, 10)
        self.agents = [Agent(0, 0, time_step), Agent(1, 1, time_step)]
        self.a_star = AStar(self.arena, self.agents)

    def test_a_star_init(self):
        self.assertEqual(self.arena, self.a_star.arena)
        self.assertEqual(self.agents, self.a_star.agents)
        self.assertEqual([], self.a_star.path)

    def test_check_blocked_square_fails(self):
        self.arena.set_blockage([0], [1])
        status = self.a_star.check_target_location(0, 1)
        self.assertEqual(RoutingStatus.TARGET_BLOCKED, status)

    def test_check_reserved_square_fails(self):
        self.arena.set_reserved([0], [1])
        status = self.a_star.check_target_location(0, 1)
        self.assertEqual(RoutingStatus.TARGET_RESERVED, status)

    def test_route_to_blocked_square_fails(self):
        self.arena.set_blockage([0], [1])
        status = self.a_star.route(self.agents[0], (0, 1))
        self.assertEqual(RoutingStatus.TARGET_BLOCKED, status)

    def test_route_to_reserved_square_fails(self):
        self.arena.set_reserved([0], [1])
        status = self.a_star.route(self.agents[0], (0, 1))
        self.assertEqual(RoutingStatus.TARGET_RESERVED, status)

    def test_generate_nodes(self):
        start_node = Node((0, 0))
        end_node = Node((0, 4))
        target = (0, 4)
        self.a_star.generate_nodes(self.agents[0], target)
        start_eq = start_node == self.a_star.start
        end_eq = end_node == self.a_star.target
        self.assertTrue(start_eq)
        self.assertTrue(end_eq)
