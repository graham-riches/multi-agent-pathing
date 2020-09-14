"""
    @file a_star.py
    @brief A* path finding algorithm
    @author Graham Riches
    @details
    A* path finding algorithm implementation.
    A* seeks to minimize a cost function for a route by calculating the cost value of the path at each node.
    For each route tree, the nodes store their parent node, which results in the path being solved by tracing the
    lowest cost path from end to start.
"""

from arena import Arena
from agent import Agent
from routing.routing_algorithm import Algorithm
from routing.status import RoutingStatus
from tile import TileState


class Node:
    def __init__(self, location: tuple, parent=None) -> None:
        """
        Generate a node at a specific location.
        :param location: the location tuple (x,y)
        :param parent: A parent Node. Note, I would normally use typehints here, but typehints don't play well
                       with self-references and using things like typings Optional to simulate C++ templates seems
                       kind of pointless.
        """
        self.location = location
        self.parent = parent
        # store the value of any algorithm weights
        self.total_cost = 0
        self.travelled_cost = 0
        self.heuristic_cost = 0

    def __eq__(self, other) -> bool:
        """
        compare two Node objects
        :param other: another node
        """
        return self.location == other.location


class AStar(Algorithm):
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Initialize an A* search object given a particular arena and a list of agents. Agent locations are considered
        as blockages that must be handled in the search
        """
        super(AStar, self).__init__(arena, agents)
        self.start = None
        self.target = None

    def route(self, agent: Agent, target: tuple) -> RoutingStatus:
        """
        Main routing method override from abstract base class. This calls the functionality to route an Agent
        from one location to another
        :param agent: the agent to route
        :param target: the ending location tuple (x,y)
        :return: RoutingStatus
        """
        # check that the target location is valid
        status = self.check_target_location(target[0], target[1])
        if status is not RoutingStatus.SUCCESS:
            return status

        # generate the start and target nodes
        self.generate_nodes(agent, target)

        # start generating the list of paths

        # continually add paths until the target node is reached

        return status

    def generate_nodes(self, agent: Agent, target: tuple) -> None:
        """
        generate the starting and ending nodes
        :param agent: the agent
        :param target: the target location tuple (x,y)
        :return: None
        """
        self.start = Node((agent.location.X, agent.location.Y))
        self.target = Node(target)

    def check_target_location(self, x: int, y: int) -> RoutingStatus:
        """
        Check if the target location is valid
        :param x: the x coordinate of the target
        :param y: the y coordinate of the target
        :return:
        """
        tile_state = self.arena.get_tile_state(x, y)
        if tile_state == TileState.BLOCKED:
            return RoutingStatus.TARGET_BLOCKED
        elif tile_state == TileState.RESERVED:
            return RoutingStatus.TARGET_RESERVED
        else:
            return RoutingStatus.SUCCESS

