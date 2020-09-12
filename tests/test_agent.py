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

    def test_set_acceleration(self):
        self.agent.acceleration = 3
        self.assertEqual(3, self.agent.acceleration)

    def test_set_velocity(self):
        self.agent.velocity = 3
        self.assertEqual(3, self.agent.velocity)

    def test_set_deceleration(self):
        self.agent.deceleration = 3
        self.assertEqual(3, self.agent.deceleration)

    def test_triangular_profile(self) -> None:
        motion_profile = self.agent.generate_motion_profile(4, 0.005)
        self.assertEqual(4, motion_profile[-1])
        # Note: this length is based on some sketchy testing, but it's roughly correct
        self.assertEqual(693, len(motion_profile))

    def test_trapezoidal_profile(self) -> None:
        motion_profile = self.agent.generate_motion_profile(8, 0.001)
        self.assertEqual(8, motion_profile[-1])
        # Note: this length is based on some sketchy testing, but it's roughly correct
        self.assertEqual(4917, len(motion_profile))

