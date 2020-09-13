"""
    @file test_command_line.py
    @brief unit tests for the CLI
    @author Graham Riches
    @details
   
"""
import unittest
from command_line import CommandLine
from agent import Agent, AgentCoordinates
from arena import Arena


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        time_step = 0.05
        agents = [Agent(0, 0, time_step)]
        arena = Arena(10, 20)
        self.cli = CommandLine(arena, agents)

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
        command = 'blockage 4 4'
        retval = self.cli.parse_command(command)
        self.assertTrue(retval)
