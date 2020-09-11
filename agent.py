"""
    @file agent.py
    @brief single agent object
    @author Graham Riches
    @details
    Single simulation agent that can move a predetermined amount with a given kinematic profile.
"""
import numpy as np


class Agent:
    def __init__(self, acceleration: float, deceleration: float, velocity: float) -> None:
        """
        Create an agent object with a given kinematic move profile. Note: the units of the profile are
        completely arbitrary but are assumed to have the same base units (i.e. meters) as any movement distance.
        Note: consider acceleration, deceleration, and velocity as scalar quantities and let the chosen move
              direction convert everything into vector quantities
        :param acceleration: the agents acceleration
        :param deceleration: the agents deceleration
        :param velocity: the agents velocity
        """
        self._acceleration = acceleration
        self._deceleration = deceleration
        self._velocity = velocity

    def generate_motion_profile(self, distance: float, samples: int) -> np.array:
        """
        Generates a motion profile with a given number of sample points
        :param distance: how far to move the agent in "agent units"
        :param samples: how many samples to use in the profile
        :return: list containing the positions at each sample interval
        """
        # generate some lists of the vectors to fill
        accel_profile = np.ones(samples)
        vel_profile = np.ones(samples)
        pos_profile = np.ones(samples)

        # get the move direction
        

        # calculate how long it takes the agent to get up to speed and to stop
        accel_time = self._velocity/self._acceleration
        decel_time = self._velocity/self._deceleration

        # calculate how far the agent travels during accel and decel
        accel_dist = 0.5 * self._acceleration * accel_time**2
        decel_dist = 0.5 * self._deceleration * decel_time**2

        if (accel_dist + decel_dist) >= abs(distance):
            # motion profile is triangular
            accel_actual_dist = distance * accel_dist / (accel_dist + decel_dist)
            decel_actual_dist = distance * decel_dist / (accel_dist + decel_dist)

            # determine the time that is spent accelerating and decelerating
            accel_actual_time = (2 * accel_actual_dist / self._acceleration)**0.5
            decel_actual_time = (2 * decel_actual_dist / self._deceleration)**0.5

            # determine how samples are accelerating and how many are decelerating
            accel_done_sample = round(samples*accel_actual_time / (accel_actual_time + decel_actual_time))
            accel_profile[0:accel_done_sample] =




        else:
            pass
        return pos_profile



