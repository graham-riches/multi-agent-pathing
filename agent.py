"""
    @file agent.py
    @brief single agent object
    @author Graham Riches
    @details
    Single simulation agent that can move a predetermined amount with a given kinematic profile.
"""
import numpy as np


class Agent:
    def __init__(self, acceleration: float = 2.0, deceleration: float = 2.0, velocity: float = 1.0) -> None:
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

    @property
    def acceleration(self) -> float:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration: float) -> None:
        self._acceleration = acceleration

    @property
    def velocity(self) -> float:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: float) -> None:
        self._velocity = velocity

    @property
    def deceleration(self) -> float:
        return self._deceleration

    @deceleration.setter
    def deceleration(self, deceleration: float) -> None:
        self._deceleration = deceleration

    def generate_motion_profile(self, distance: float, timestep: float) -> np.array:
        """
        Generates a motion profile with a given number of sample points
        :param distance: how far to move the agent in "agent units"
        :param timestep: simulation timestep (s)
        :return: list containing the positions at each sample interval
        """
        # get the move direction
        move_dir = np.sign(distance)

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
            total_time = accel_actual_time + decel_actual_time
            samples = round(total_time / timestep)
            time_vector = np.linspace(0, total_time, samples)
            accel_profile = np.ones(samples)

            # determine how samples are accelerating and how many are decelerating
            accel_done_sample = round(samples*accel_actual_time / total_time)
            accel_profile[0:accel_done_sample] = move_dir * self._acceleration
            accel_profile[accel_done_sample:-1] = -1 * move_dir * self._deceleration

        else:
            # determine the amount of time at constant velocity
            const_vel_dist = distance - accel_dist - decel_dist
            const_vel_time = const_vel_dist / self._velocity
            total_time = const_vel_time + accel_time + decel_time

            # create the acceleration profile
            samples = round(total_time / timestep)
            time_vector = np.linspace(0, total_time, samples)
            accel_profile = np.zeros(samples)
            accel_samples = round(samples * accel_time / total_time)
            decel_samples = round(samples * decel_time / total_time)
            const_vel_samples = samples - (accel_samples + decel_samples)
            accel_profile[0:accel_samples] = move_dir * self._acceleration
            accel_profile[-decel_samples:-1] = -1 * move_dir * self._deceleration
            pass

        # generate empty position and velocity vectors
        vel_profile = np.zeros(samples)
        pos_profile = np.zeros(samples)

        # dt may not be exactly the same as the timestep so re-calculate it
        dt = time_vector[1]

        # integrate the acceleration profile twice to get the velocity
        for i in range(1, samples):
            vel_profile[i] = vel_profile[i-1] + dt*accel_profile[i]
        for i in range(1, samples):
            pos_profile[i] = pos_profile[i-1] + dt*vel_profile[i]
        # remove the integration error by fudging the last value
        pos_profile[-1] = distance
        return pos_profile



