"""
    @file
    @brief
    @author
    @details
   
"""
from unittest import TestCase
from agent import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        self.agent = Agent(2, 1, 3)

    def test_triangular_profile(self) -> None:
        self.agent.generate_motion_profile(4, 100)

