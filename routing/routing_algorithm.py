"""
    @file routing_algorithm.py
    @brief abstract base class for various routing algorithms
    @author Graham Riches
    @details
    Abstract base class for a routing algorithm. This lets the routing manager accept any time of routing algorithm
    as long as it supplies specific methods.
"""

from abc import ABC, abstractmethod
from arena import Arena
from agent import *
from routing.status import RoutingStatus


class SingleAgentAlgorithm(ABC):
    @abstractmethod
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Initialize a routing algorithm with the Arena and a list of agents. This finds an "optimal" path to
        a goal for a single agent using whatever means the child classes chooses.
        :param arena: the arena for the simulation
        :param agents: the list of all simulation agents
        """
        self.arena = arena
        self.agents = agents
        self.node_path = list()  # contains all the nodes that are part of a target route
        self.path = list()  # contains a list of agent tasks to create the route

    @abstractmethod
    def route(self, agent: Agent, target: tuple) -> RoutingStatus:
        """
        Abstract routing method that any algorithm can implement to do a custom route from
        start to target.
        :param agent: agent to route
        :param target: ending location tuple (x,y)
        :return: RoutingStatus enumeration
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the routing algorithm to clear any internal state variables
        :return:
        """
        pass

    def create_path(self) -> RoutingStatus:
        """
        Traverses a list of nodes that compose the path's node_path and constructs a list of agent
        tasks required to travel the path
        :return:
        """
        if len(self.node_path) == 0:
            return RoutingStatus.INVALID_PATH

        while len(self.node_path) > 1:
            # initialize the first path and direction
            task_start_node = self.node_path.pop(0)
            last_node = self.node_path[0]
            if last_node.location[0] == task_start_node.location[0]:
                task_direction = AgentCoordinates.Y
            elif last_node.location[1] == task_start_node.location[1]:
                task_direction = AgentCoordinates.X
            else:
                return RoutingStatus.INVALID_PATH

            # traverse the nodes until we see a turn
            nodes = list(self.node_path)
            pop_count = 0
            for next_node in nodes:
                if task_direction == AgentCoordinates.Y:
                    if next_node.location[0] != last_node.location[0]:
                        break
                    else:
                        last_node = next_node
                else:
                    if next_node.location[1] != last_node.location[1]:
                        break
                    else:
                        last_node = next_node
                pop_count += 1

            # pop everything up until the turn (current last index - 1)
            while pop_count > 1:
                self.node_path.pop(0)
                pop_count -= 1

            # create the task and add it to the path list
            if task_direction == AgentCoordinates.X:
                move_distance = last_node.location[0] - task_start_node.location[0]
            else:
                move_distance = last_node.location[1] - task_start_node.location[1]
            self.path.append(AgentTask(AgentTasks.MOVE, [task_direction, move_distance]))
        return RoutingStatus.SUCCESS


class MultiAgentAlgorithm(ABC):
    @abstractmethod
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Creates a multi-agent routing algorithm. This manages routing a group of agents towards a goal.
        :param arena:
        :param agents:
        """
        self.arena = arena  # the arena object
        self.agents = agents  # list of agents
        self.agent_goals = [None for agent in self.agents]  # goal location for each agent

    def set_agent_goal(self, agent_id: int, location: tuple) -> None:
        """
        set the goal location for an agent. The algorithm will continually route to here until 'Done'
        :param agent_id: the id of the agent
        :param location: the target/goal location
        :return: None
        """
        self.agent_goals[agent_id] = location
