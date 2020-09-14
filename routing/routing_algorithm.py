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
from agent import Agent
from routing.status import RoutingStatus


class Algorithm(ABC):
    @abstractmethod
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Initialize a routing algorithm with the Arena and a list of agents
        :param arena: the arena for the simulation
        :param agents: the list of all simulation agents
        """
        self.arena = arena
        self.agents = agents
        # path list will contain the chosen route from start to end as a set of agent tasks
        self.path = list()

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
