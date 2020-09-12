"""
    @file agent.py
    @brief single agent object
    @author Graham Riches
    @details
    Single simulation agent that can move a predetermined amount with a given kinematic profile.
    An Agent has kinematic properties such as acceleration, deceleration, velocity, etc. and can move
    to a specific target position given those parameters. An agent stores it's motion profile internally and
    will update it's location whenever it's update method is called.
"""
import numpy as np
from enum import Enum
from agent_exceptions import MotionError


class AgentCoordinates(Enum):
    X = 1
    Y = 2


class AgentState(Enum):
    IDLE = 0
    MOVING = 1


class AgentLocation:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        """
        Location manager object to keep track of an agents X any Y location
        :param: x: x location
        :param: y: y location
        """
        self._x = x
        self._y = y

    @property
    def X(self) -> int:
        return self._x

    @X.setter
    def X(self, position: int) -> None:
        self._x = position

    @property
    def Y(self):
        return self._y

    @Y.setter
    def Y(self, position: int) -> None:
        self._y = position

    def update(self, direction: AgentCoordinates, update_position: float) -> None:
        """
        update an AgentLocation object in a specific direction
        :param direction: which direction to update
        :param update_position: the position update. This is added to the agents current location
        :return: None
        """
        if direction == AgentCoordinates.X:
            self._x += update_position
        elif direction == AgentCoordinates.Y:
            self._y += update_position


class AgentMotionProfile:
    def __init__(self, acceleration: float = 2.0, deceleration: float = 2.0, velocity: float = 1.0) -> None:
        """
        Create an agent motion profile object with a given kinematic move profile. Note: the units of the profile are
        completely arbitrary but are assumed to have the same base units (i.e. meters) as any movement distance.
        Note: consider acceleration, deceleration, and velocity as scalar quantities and let the chosen move
              direction convert everything into vector quantities
        :param acceleration: the agents acceleration
        :param deceleration: the agents deceleration
        :param velocity: the agents velocity
        """
        # kinematic parameters
        self._acceleration = acceleration
        self._deceleration = deceleration
        self._velocity = velocity
        # profiles
        self.position_profile = None
        self.time_vector = None
        # helpers
        self._accel_time = 0
        self._decel_time = 0
        self._accel_dist = 0
        self._decel_dist = 0
        self.__calculate_acceleration_values()

    @property
    def acceleration(self) -> float:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration: float) -> None:
        self._acceleration = abs(acceleration)
        self.__calculate_acceleration_values()

    @property
    def velocity(self) -> float:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: float) -> None:
        self._velocity = abs(velocity)
        self.__calculate_acceleration_values()

    @property
    def deceleration(self) -> float:
        return self._deceleration

    @deceleration.setter
    def deceleration(self, deceleration: float) -> None:
        self._deceleration = abs(deceleration)
        self.__calculate_acceleration_values()

    @property
    def in_motion(self) -> bool:
        return self._in_motion

    def __calculate_acceleration_values(self) -> None:
        """
        Private method: Update the agents internal variables
        :return: None
        """
        self._accel_time = self._velocity/self._acceleration
        self._decel_time = self._velocity/self._deceleration
        self._accel_dist = 0.5 * self._acceleration * self._accel_time**2
        self._decel_dist = 0.5 * self._deceleration * self._decel_time ** 2

    def __is_move_triangular(self, distance: float) -> bool:
        """
        Private method: checks if a move profile is triangular or trapezoidal
        :param distance:
        :return: bool True if triangular
        """
        return (self._accel_dist + self._decel_dist) >= abs(distance)

    def generate_motion_profile(self, distance: float, timestep: float) -> int:
        """
        Generates a motion profile with a given number of sample points
        :param distance: how far to move the agent in "agent units"
        :param timestep: simulation timestep (s)
        :return: how many timesteps it takes to complete the move
        """
        # get the move direction
        move_dir = np.sign(distance)

        if self.__is_move_triangular(distance):
            # motion profile is triangular
            accel_actual_dist = distance * self._accel_dist / (self._accel_dist + self._decel_dist)
            decel_actual_dist = distance * self._decel_dist / (self._accel_dist + self._decel_dist)

            # determine the time that is spent accelerating and decelerating
            accel_actual_time = (2 * accel_actual_dist / self._acceleration)**0.5
            decel_actual_time = (2 * decel_actual_dist / self._deceleration)**0.5
            total_time = accel_actual_time + decel_actual_time
            samples = round(total_time / timestep)
            self.time_vector = np.linspace(0, total_time, samples)
            accel_profile = np.ones(samples)

            # determine how samples are accelerating and how many are decelerating
            accel_done_sample = round(samples*accel_actual_time / total_time)
            accel_profile[0:accel_done_sample] = move_dir * self._acceleration
            accel_profile[accel_done_sample:-1] = -1 * move_dir * self._deceleration

        else:
            # determine the amount of time at constant velocity
            const_vel_dist = distance - self._accel_dist - self._decel_dist
            const_vel_time = const_vel_dist / self._velocity
            total_time = const_vel_time + self._accel_time + self._decel_time

            # create the acceleration profile
            samples = round(total_time / timestep)
            self.time_vector = np.linspace(0, total_time, samples)
            accel_profile = np.zeros(samples)
            accel_samples = round(samples * self._accel_time / total_time)
            decel_samples = round(samples * self._decel_time / total_time)
            const_vel_samples = samples - (accel_samples + decel_samples)
            accel_profile[0:accel_samples] = move_dir * self._acceleration
            accel_profile[-decel_samples:-1] = -1 * move_dir * self._deceleration

        # generate empty position and velocity vectors
        vel_profile = np.zeros(samples)
        self.position_profile = np.zeros(samples)

        # dt may not be exactly the same as the time-step so re-calculate it
        dt = self.time_vector[1]

        # integrate the acceleration profile twice to get the velocity
        for i in range(1, samples):
            vel_profile[i] = vel_profile[i-1] + dt*accel_profile[i]
        for i in range(1, samples):
            self.position_profile[i] = self.position_profile[i - 1] + dt * vel_profile[i]

        # remove the integration error by fudging the last value
        self.position_profile[-1] = distance
        return samples


class Agent:
    def __init__(self, x_position: int, y_position: int, simulation_time_step: float) -> None:
        """
        Spawns an Agent object that can move about in the coordinate system defined in AgentCoordinates at initial
        location given by the input coordinates.
        The agents location is managed by an AgentLocation object and it's motion parameters are calculated
        using an AgentMotionProfile.
        This object serves as an interface to a single Agent that has these properties
        :param x_position: Agents spawned x location
        :param y_position: Agents spawned y location
        :param simulation_time_step: base time step of the simulation for the agent object
        """
        # public
        self.location = AgentLocation(x_position, y_position)
        self.state = AgentState.IDLE
        # private
        self._time_step = simulation_time_step
        self._movement_steps = None
        self._current_time_step = 0
        self._motion_profile = AgentMotionProfile()  # use default motion parameters
        self._current_direction = None

    def set_kinematic_parameters(self, acceleration: float, deceleration: float, velocity: float) -> None:
        """
        Wrapper function to set an Agents kinematic properties. Note: this can only occur while the agent is stationary
        :param acceleration: agent acceleration in "position" units
        :param deceleration: agent deceleration in "position" units
        :param velocity: agent velocity in "position" units
        :return: None
        """
        if self.state == AgentState.MOVING:
            raise MotionError('Cannot set agent parameters while in motion')
        self._motion_profile.acceleration = acceleration
        self._motion_profile.deceleration = deceleration
        self._motion_profile.velocity = velocity

    def start_move(self, direction: AgentCoordinates, distance: float) -> None:
        """
        Start an agent on a move in a specific direction
        :param direction: coordinate direction to move
        :param distance: distance to move
        :return: True if started, false if there is an error
        """
        if self.state == AgentState.MOVING:
            raise MotionError('Agent is already in motion')
        # generate the agents motion profile:
        self._movement_steps = self._motion_profile.generate_motion_profile(distance, self._time_step)
        self._current_direction = direction
        self.state = AgentState.MOVING
        # reset the current time step
        self._current_time_step = 0
        # set the agent's initial position
        self.location.update(self._current_direction, self._motion_profile.position_profile[self._current_time_step])

    def update(self) -> AgentState:
        """
        Method called every base time step to update the agent object
        :return: the agents current state. Higher level simulation can trigger an update when state flips to IDLE
        """
        self._current_time_step += 1
        self.location.update(self._current_direction, self._motion_profile.position_profile[self._current_time_step])

        # check if the movement is complete
        if self._current_time_step == self._movement_steps-1:
            self.state = AgentState.IDLE
        return self.state



