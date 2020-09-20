"""
    @file test_a_star.py
    @brief unit tests for A* path finding
    @author Graham Riches
    @details
   
"""
import unittest
from arena import Arena
from agent import *
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

    def test_initialize_nodes(self):
        start = (0, 0)
        target = (0, 4)
        start_node = Node(start)
        end_node = Node(target)
        self.a_star.initialize_nodes(self.agents[0], target)
        start_eq = start_node == self.a_star.start
        end_eq = end_node == self.a_star.target
        self.assertTrue(start_eq)
        self.assertTrue(end_eq)

    def test_get_new_nodes_in_empty_area(self):
        # initialize the search, and then get neighbouring tiles
        start = (1, 1)
        expected_neighbours = [(2, 1), (1, 2), (0, 1), (1, 0)]
        parent_node = Node(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = Node(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_get_new_nodes_in_corner(self):
        # initialize the search in a corner, and then get neighbouring tiles
        start = (0, 0)
        expected_neighbours = [(0, 1), (1, 0)]
        parent_node = Node(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = Node(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_get_new_nodes_with_blockage(self):
        # initialize the search, set some blockages, and then make sure only the free tiles are returned
        start = (1, 1)
        self.arena.set_blockage([2], [0, 1, 2, 3])
        expected_neighbours = [(1, 2), (0, 1), (1, 0)]
        parent_node = Node(start)
        new_nodes = self.a_star.generate_new_nodes(parent_node)
        self.assertEqual(len(expected_neighbours), len(new_nodes))
        # make sure all of the correct nodes are returned
        for node_location in expected_neighbours:
            test_node = Node(node_location)
            node_exists = test_node in new_nodes
            self.assertTrue(node_exists)

    def test_new_node_with_parent_has_a_higher_cost(self):
        start = (1, 1)
        start_node = Node(start)
        new_node = Node((2, 1), parent=start_node)
        self.assertEqual(1, new_node.travelled_cost)

    def test_calculate_heuristic(self):
        start = (1, 1)
        target = (4, 4)
        start_node = Node(start)
        target_node = Node(target)
        start_node.calculate_heuristic(target_node)
        self.assertEqual(18, start_node.heuristic_cost)

    def test_routing_simple_path(self):
        target = (4, 4)
        status = self.a_star.route(self.agents[0], target)
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # make sure the final nodes are correct
        self.assertTupleEqual((4, 4), self.a_star.node_path[-1].location)

    def test_reconstruct_empty_path_is_invalid(self):
        self.a_star.node_path = list()
        status = self.a_star.create_path()
        self.assertEqual(RoutingStatus.INVALID_PATH, status)

    def test_diagonal_path_is_invalid(self):
        self.a_star.node_path = [Node((0, 0)), Node((1, 1))]
        status = self.a_star.create_path()
        self.assertEqual(RoutingStatus.INVALID_PATH, status)

    def test_reconstruct_path(self):
        # force a list of nodes to be the node path
        self.a_star.node_path = [Node((0, 0)), Node((0, 1)), Node((0, 2)), Node((1, 2)), Node((2, 2))]
        status = self.a_star.create_path()
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # The order of the tasks should be: Move X 2, Move Y 2
        task_direction = [AgentCoordinates.Y, AgentCoordinates.X]
        task_distance = [2, 2]
        for idx, task in enumerate(self.a_star.path):
            self.assertEqual(task_direction[idx], task.args[0])
            self.assertEqual(task_distance[idx], task.args[1])

    def test_reconstruct_path_zig_zag(self):
        # force a list of nodes to be the node path
        self.a_star.node_path = [Node((0, 0)), Node((1, 0)), Node((1, 1)), Node((2, 1)), Node((2, 2)),
                                 Node((3, 2)), Node((3, 3)), Node((4, 3)), Node((4, 4))]
        status = self.a_star.create_path()
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # this path should contain 8 tasks
        self.assertEqual(8, len(self.a_star.path))
        # The order of the tasks should be: Move X 1, Move Y 1, Move X 1, Move Y 1
        task_direction = [AgentCoordinates.X, AgentCoordinates.Y, AgentCoordinates.X, AgentCoordinates.Y,
                          AgentCoordinates.X, AgentCoordinates.Y, AgentCoordinates.X, AgentCoordinates.Y]
        task_distance = [1, 1, 1, 1, 1, 1, 1, 1]
        for idx, task in enumerate(self.a_star.path):
            print(task.args[0], task.args[1])
            self.assertEqual(task_direction[idx], task.args[0])
            self.assertEqual(task_distance[idx], task.args[1])
