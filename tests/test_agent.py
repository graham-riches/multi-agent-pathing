"""
    @file
    @brief
    @author
    @details
   
"""
import unittest
from agent import *
from agent_exceptions import MotionError

BASE_TIME_STEP = 0.005


class TestAgentMotionProfile(unittest.TestCase):
    def setUp(self):
        self.motion_profile = AgentMotionProfile(2, 1, 3)

    def test_agent_init(self):
        self.assertEqual(2, self.motion_profile.acceleration)
        self.assertEqual(1, self.motion_profile.deceleration)
        self.assertEqual(3, self.motion_profile.velocity)

    def test_set_acceleration(self):
        self.motion_profile.acceleration = 3
        self.assertEqual(3, self.motion_profile.acceleration)
        self.motion_profile.acceleration = -3
        self.assertEqual(3, self.motion_profile.acceleration)

    def test_set_velocity(self):
        self.motion_profile.velocity = 3
        self.assertEqual(3, self.motion_profile.velocity)
        self.motion_profile.velocity = -3
        self.assertEqual(3, self.motion_profile.velocity)

    def test_set_deceleration(self):
        self.motion_profile.deceleration = 3
        self.assertEqual(3, self.motion_profile.deceleration)
        self.motion_profile.deceleration = -3
        self.assertEqual(3, self.motion_profile.deceleration)

    def test_triangular_profile(self):
        profile_length = self.motion_profile.generate_motion_profile(4, 0.005)
        # Note: this length is based on some sketchy testing, but it's roughly correct
        self.assertEqual(693, profile_length)

    def test_trapezoidal_profile(self):
        profile_length = self.motion_profile.generate_motion_profile(8, 0.001)
        # Note: this length is based on some sketchy testing, but it's roughly correct
        self.assertEqual(4917, profile_length)

    def test_negative_triangular_profile(self):
        profile_length = self.motion_profile.generate_motion_profile(-4, 0.005)
        # Note: this length is based on some sketchy testing, but it's roughly correct
        self.assertEqual(693, profile_length)


class TestAgentLocation(unittest.TestCase):
    def setUp(self):
        self.location = AgentLocation()

    def test_location_init(self):
        self.assertEqual(0, self.location.X)
        self.assertEqual(0, self.location.Y)

    def test_agent_update(self):
        self.location.update(AgentCoordinates.X, 0.5)
        self.assertEqual(0.5, self.location.X)
        self.location.update(AgentCoordinates.Y, 0.5)
        self.assertEqual(0.5, self.location.Y)
        self.location.update(AgentCoordinates.X, -3)
        self.assertEqual(-3, self.location.X)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(0, 0, BASE_TIME_STEP)

    def test_agent_init(self):
        self.assertEqual(AgentState.IDLE, self.agent.state)

    def test_setting_profile_while_stopped_works(self):
        try:
            self.agent.set_kinematic_parameters(2, 1, 3)
        except MotionError:
            self.fail('Raised unexpected exception')

    def test_setting_profile_while_in_motion_raises_exception(self):
        self.agent.start_move(AgentCoordinates.X, 4)
        with self.assertRaises(MotionError):
            self.agent.set_kinematic_parameters(2, 1, 3)

    def test_running_agent_sim_returns_idle_when_motion_completes(self):
        self.agent.start_move(AgentCoordinates.X, 4)
        # get a private member for testing
        sim_steps = self.agent._movement_steps
        for i in range(sim_steps-1):
            agent_state = self.agent.update()
        self.assertEqual(AgentState.IDLE, agent_state)
