"""
    @file test_a_star.py
    @brief unit tests for A* path finding
    @author Graham Riches
    @details
   
"""
import unittest
from core.arena import Arena
from core.agent import *
from routing.a_star import AStar, AStarNode
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

    def test_initialize_nodes(self):
        start = (0, 0)
        target = (0, 4)
        start_node = AStarNode(start)
        end_node = AStarNode(target)
        self.a_star.initialize_nodes(self.agents[0], target)
        start_eq = start_node == self.a_star.start
        end_eq = end_node == self.a_star.target
        self.assertTrue(start_eq)
        self.assertTrue(end_eq)

    def test_get_new_nodes_in_empty_area(self):
        # initialize the search, and then get neighbouring tiles
        start = (1, 1)
        expected_neighbours = [(2, 1), (1, 2), (0, 1), (1, 0)]
        parent_node = AStarNode(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = AStarNode(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_get_new_nodes_in_corner(self):
        # initialize the search in a corner, and then get neighbouring tiles
        start = (0, 0)
        expected_neighbours = [(0, 1), (1, 0)]
        parent_node = AStarNode(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = AStarNode(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_get_new_nodes_with_blockage(self):
        # initialize the search, set some blockages, and then make sure only the free tiles are returned
        start = (1, 1)
        self.arena.set_blockage([2], [0, 1, 2, 3])
        expected_neighbours = [(1, 2), (0, 1), (1, 0)]
        parent_node = AStarNode(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = AStarNode(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_calculate_heuristic(self):
        start = (1, 1)
        target = (4, 4)
        start_node = AStarNode(start)
        target_node = AStarNode(target)
        heuristic = self.a_star.calculate_heuristic_cost(start_node, target_node)
        self.assertEqual(6, heuristic)

    @unittest.skip('debug')
    def test_routing_simple_path(self):
        target = (4, 4)
        status = self.a_star.route(self.agents[0], target)
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # make sure the final nodes are correct
        self.assertTupleEqual((4, 4), self.a_star.node_path[-1].location)

    def test_route_to_tile_with_other_agent_fails(self):
        status = self.a_star.check_target_location(1, 1)
        self.assertEqual(RoutingStatus.TARGET_RESERVED, status)

    def test_calculate_turn_cost_no_factor(self):
        start_node = AStarNode((0, 0))
        node_1 = AStarNode((1, 0), parent=start_node)
        node_2 = AStarNode((2, 0), parent=node_1)
        node_3 = AStarNode((2, 1), parent=node_2)
        node_4 = AStarNode((2, 2), parent=node_3)
        node_5 = AStarNode((3, 2), parent=node_4)
        turn_cost = self.a_star.calculate_turn_cost(node_5)
        self.assertEqual(0, turn_cost)

    def test_calculate_turn_cost_with_weight(self):
        start_node = AStarNode((0, 0))
        self.a_star.turn_factor = 1
        node_1 = AStarNode((1, 0), parent=start_node)
        node_2 = AStarNode((2, 0), parent=node_1)
        node_3 = AStarNode((2, 1), parent=node_2)
        node_4 = AStarNode((2, 2), parent=node_3)
        node_5 = AStarNode((3, 2), parent=node_4)
        turn_cost = self.a_star.calculate_turn_cost(node_5)
        self.assertEqual(2, turn_cost)

    def test_calculate_inline_cost(self):
        self.a_star.inline_factor = 10
        self.arena.set_agent_target(3, 3)
        node = AStarNode((1, 3))
        score = self.a_star.calculate_inline_cost(node)
        self.assertEqual(10, score)
