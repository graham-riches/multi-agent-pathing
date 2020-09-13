"""
    @file test_routing_manager.py
    @brief test the routing manager functionality
    @author Graham Riches
    @details
   
"""
import unittest
from agent import *
from arena import *
from routing.routing_manager import RoutingManager, AgentEvent


class TestRoutingManager(unittest.TestCase):
    def setUp(self):
        time_step = 0.005
        self.agent = Agent(0, 0, time_step)
        self.agents = [self.agent]
        self.arena = Arena(10, 10)
        self.routing_manager = RoutingManager(self.arena, self.agents)

    def test_signal_agent_task_completed(self):
        self.routing_manager.signal_agent_event(0, AgentEvent.TASK_COMPLETED)

    def test_queing_and_popping_agent_tasks(self):
        task = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 4])
        self.routing_manager.add_agent_task(0, task)
        # signaling that the agent completed a task should kick off the next task
        self.routing_manager.signal_agent_event(0, AgentEvent.TASK_COMPLETED)
        self.assertEqual(self.routing_manager.agent_tasks[0], [])
