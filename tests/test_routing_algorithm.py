"""
    @file test_routing_algorithm.py
    @brief unit tests for routing algorithm base classes
    @author Graham Riches
    @details
        Unit tests for single- and multi-agent pathing algorithms
"""
import unittest
from routing.status import RoutingStatus
from routing.routing_algorithm import SingleAgentAlgorithm, MultiAgentAlgorithm, Node
from core.arena import Arena
from core.agent import *
from core.tile import TileState


class NodeMock(Node):
    def __init__(self, location: tuple, parent=None) -> None:
        """
        mock Node object for testing
        :param location: location tuple of the node
        :param parent: parent node (optional)
        """
        super(NodeMock, self).__init__(location, parent)


class SingleAgentAlgorithmMock(SingleAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        creates a mock algorithm class for testing
        :param arena: the sim arena
        :param agents: sim agents
        """
        super(SingleAgentAlgorithmMock, self).__init__(arena, agents)

    def route(self, agent: Agent, target: tuple) -> RoutingStatus:
        """
        stub routing function
        :param agent:
        :param target:
        :return: status
        """
        return RoutingStatus.SUCCESS

    def reset(self) -> None:
        """
        stub reset function
        :return: None
        """
        pass


class MultiAgentAlgorithmMock(MultiAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list, algorithm: SingleAgentAlgorithm) -> None:
        """
        mock multi-agent algorithm for testing
        :param arena: arena object
        :param agents: list of agents
        :param algorithm: main routing algorithm
        """
        super(MultiAgentAlgorithmMock, self).__init__(arena, agents, algorithm)

    def run_time_step(self) -> None:
        pass

    def route(self, agent_id: int, location: tuple) -> None:
        pass


class TestMultiAgentAlgorithm(unittest.TestCase):
    def setUp(self):
        time_step = 0.005
        self.arena = Arena(10, 10)
        self.agents = [Agent(0, 0, time_step), Agent(5, 5, time_step)]
        self.algorithm = SingleAgentAlgorithmMock(self.arena, self.agents)
        self.routing_manager = MultiAgentAlgorithmMock(self.arena, self.agents, self.algorithm)

    def tearDown(self):
        pass

    def test_is_simulation_completed(self):
        self.routing_manager.set_agent_goal(0, (0, 0))
        self.routing_manager.set_agent_goal(1, (5, 5))
        self.assertTrue(self.routing_manager.is_simulation_complete())

    def test_routing_blockages(self):
        move_x_task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 3])
        move_y_task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.Y, 3])
        self.routing_manager.add_agent_task(0, move_x_task)
        for i in range(1, 3):
            self.assertEqual(TileState.RESERVED, self.arena.get_tile_state(i, 0))
        self.assertEqual(TileState.AGENT_TARGET, self.arena.get_tile_state(3, 0))
        self.routing_manager.add_agent_task(0, move_y_task)
        for i in range(1, 3):
            self.assertEqual(TileState.RESERVED, self.arena.get_tile_state(3, i))
        self.assertEqual(TileState.AGENT_TARGET, self.arena.get_tile_state(3, 3))

    def test_adding_and_completing_task_clears_blockage(self):
        move_x_task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 3])
        self.routing_manager.add_agent_task(0, move_x_task)
        for i in range(1, 3):
            self.assertEqual(TileState.RESERVED, self.arena.get_tile_state(i, 0))
        self.assertEqual(TileState.AGENT_TARGET, self.arena.get_tile_state(3, 0))
        self.routing_manager.signal_agent_event(0, AgentEvent.TASK_COMPLETED)
        for i in range(1, 4):
            self.assertEqual(TileState.FREE, self.arena.get_tile_state(i, 0))

    def test_completing_task_clears_agent_active(self):
        move_x_task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 3])
        self.routing_manager.add_agent_task(0, move_x_task)
        self.routing_manager.signal_agent_event(0, AgentEvent.TASK_COMPLETED)
        self.assertFalse(self.routing_manager.active_agents[0])

    def test_initialize_algorithm(self):
        self.routing_manager.initialize()
        self.assertTrue(self.routing_manager.initialized)

    def test_is_agent_at_goal(self):
        self.routing_manager.set_agent_goal(0, (0, 0))
        self.assertTrue(self.routing_manager.is_agent_at_goal(0))

    def test_start_new_task(self):
        move_x_task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 3])
        self.routing_manager.add_agent_task(0, move_x_task)
        self.routing_manager.start_new_task(0)
        self.assertTrue(self.routing_manager.active_agents[0])


class TestSingleAgentAlgorithm(unittest.TestCase):
    def setUp(self):
        time_step = 0.005
        self.arena = Arena(10, 10)
        self.agents = [Agent(0, 0, time_step), Agent(1, 1, time_step)]
        self.algorithm = SingleAgentAlgorithmMock(self.arena, self.agents)

    def tearDown(self):
        pass

    def test_reconstruct_empty_path_is_invalid(self):
        self.algorithm.node_path = list()
        status = self.algorithm.create_path()
        self.assertEqual(RoutingStatus.INVALID_PATH, status)

    def test_diagonal_path_is_invalid(self):
        self.algorithm.node_path = [NodeMock((0, 0)), NodeMock((1, 1))]
        status = self.algorithm.create_path()
        self.assertEqual(RoutingStatus.INVALID_PATH, status)

    def test_reconstruct_path(self):
        # force a list of nodes to be the node path
        self.algorithm.node_path = [NodeMock((0, 0)), NodeMock((0, 1)), NodeMock((0, 2)),
                                    NodeMock((1, 2)), NodeMock((2, 2))]
        status = self.algorithm.create_path()
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # The order of the tasks should be: Move X 2, Move Y 2
        task_direction = [AgentCoordinates.Y, AgentCoordinates.X]
        task_distance = [2, 2]
        for idx, task in enumerate(self.algorithm.path):
            self.assertEqual(task_direction[idx], task.args[0])
            self.assertEqual(task_distance[idx], task.args[1])

    def test_reconstruct_path_zig_zag(self):
        # force a list of nodes to be the node path
        self.algorithm.node_path = [NodeMock((0, 0)), NodeMock((1, 0)), NodeMock((1, 1)), NodeMock((2, 1)),
                                    NodeMock((2, 2)), NodeMock((3, 2)), NodeMock((3, 3)), NodeMock((4, 3)),
                                    NodeMock((4, 4))]
        status = self.algorithm.create_path()
        self.assertEqual(RoutingStatus.SUCCESS, status)
        # this path should contain 8 tasks
        self.assertEqual(8, len(self.algorithm.path))
        # The order of the tasks should be: Move X 1, Move Y 1, Move X 1, Move Y 1
        task_direction = [AgentCoordinates.X, AgentCoordinates.Y, AgentCoordinates.X, AgentCoordinates.Y,
                          AgentCoordinates.X, AgentCoordinates.Y, AgentCoordinates.X, AgentCoordinates.Y]
        task_distance = [1, 1, 1, 1, 1, 1, 1, 1]
        for idx, task in enumerate(self.algorithm.path):
            self.assertEqual(task_direction[idx], task.args[0])
            self.assertEqual(task_distance[idx], task.args[1])
