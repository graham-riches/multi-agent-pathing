"""
    @file test_command_line.py
    @brief unit tests for the CLI
    @author Graham Riches
    @details
   
"""
import unittest
from command_line import CommandLine
from routing.routing_manager import RoutingManager
from routing.a_star import AStar
from tile import TileState
from agent import Agent
from arena import Arena


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        time_step = 0.05
        self.agents = [Agent(0, 0, time_step)]
        self.arena = Arena(10, 20)
        self.algorithm = AStar(self.arena, self.agents)
        self.manager = RoutingManager(self.arena, self.agents, self.algorithm)
        self.cli = CommandLine(self.arena, self.agents, self.manager)

    def test_help(self):
        command = 'help'
        retval = self.cli.parse_command(command)
        self.assertTrue(retval)

    def test_help_specific(self):
        command = 'help move_agent'
        retval = self.cli.parse_command(command)
        self.assertTrue(retval)

    def test_agent_move(self):
        command = 'move_agent 0 X 4'
        retval = self.cli.parse_command(command)
        self.assertTrue(retval)

    def test_blockage(self):
        command = 'blockage set 4 4'
        retval = self.cli.parse_command(command)
        self.assertTrue(retval)
        self.assertEqual(TileState.BLOCKED, self.arena.get_tile_state(4, 4))
