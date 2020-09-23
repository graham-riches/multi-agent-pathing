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
import numpy as np
from arena import Arena
from agent import Agent, AgentCoordinates
from routing.routing_algorithm import SingleAgentAlgorithm
from routing.status import RoutingStatus
from tile import TileState

TRAVELLED_COST_INITIAL = 1000000


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
        # store the value of any algorithm weights. Note: the travelled cost is directly related to the parent node
        self.total_cost = 0
        self._travelled_cost = 0
        self._heuristic_cost = 0
        self._turn_cost = 0

    @property
    def travelled_cost(self) -> float:
        return self._travelled_cost

    @travelled_cost.setter
    def travelled_cost(self, cost: float) -> None:
        self._travelled_cost = cost
        self.calculate_total_cost()

    @property
    def heuristic_cost(self) -> float:
        return self._heuristic_cost

    @heuristic_cost.setter
    def heuristic_cost(self, cost: float) -> None:
        self._heuristic_cost = cost
        self.calculate_total_cost()

    @property
    def turn_cost(self) -> float:
        return self._turn_cost

    @turn_cost.setter
    def turn_cost(self, cost: float) -> None:
        self._turn_cost = cost
        self.calculate_total_cost()

    def calculate_total_cost(self) -> None:
        """
        Calculate the total routing cost for a node
        :return:
        """
        self.total_cost = self._heuristic_cost + self._travelled_cost + self._turn_cost

    def __eq__(self, other) -> bool:
        """
        compare two Node objects
        :param other: another node
        """
        return self.location == other.location


class AStar(SingleAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Initialize an A* search object given a particular arena and a list of agents. Agent locations are considered
        as blockages that must be handled in the search
        """
        self.arena = arena
        self.agents = agents

        # initialize parent class
        super(AStar, self).__init__(self.arena, self.agents)
        # initialize routing components
        self.start = None
        self.target = None
        self.came_from = None
        self._turn_factor = 0
        self.reset()

    @property
    def turn_factor(self) -> float:
        return self._turn_factor

    @turn_factor.setter
    def turn_factor(self, factor: float) -> None:
        self._turn_factor = factor

    def reset(self) -> None:
        """
        Reset the algorithms internal properties
        :return:
        """
        # re-initialize the algorithm parent class
        super(AStar, self).__init__(self.arena, self.agents)
        self.start = None
        self.target = None
        self.came_from = None

    def route(self, agent: Agent, target: tuple) -> RoutingStatus:
        """
        Main routing method override from abstract base class. This calls the functionality to route an Agent
        from one location to another
        :param agent: the agent to route
        :param target: the ending location tuple (x,y)
        :return: RoutingStatus
        """
        # create an open set to store the current search nodes
        open_set = list()

        # create a grid to store the current path
        size_x, size_y = self.arena.get_dimensions()
        self.came_from = np.ndarray(shape=(size_x, size_y), dtype=object)  # 2D array of nodes

        # check that the target location is valid
        status = self.check_target_location(target[0], target[1])
        if status is not RoutingStatus.SUCCESS:
            return status

        # generate the start and target nodes
        self.initialize_nodes(agent, target)

        # add the start node to the open set
        open_set.append(self.start)

        # run the pathing algorithm to completion
        while len(open_set) > 0:
            # get the node with the lowest total score for the next iteration
            node_scores = [node.total_cost for node in open_set]
            index = np.argmin(node_scores)
            current_node = open_set.pop(int(index))
            # check if the current node is equal to the target and return if true
            if current_node == self.target:
                status = RoutingStatus.SUCCESS
                self.node_path = self.construct_node_path(self.target)
                self.create_path()
                break

            neighbours = self.generate_new_nodes(current_node)
            for neighbour in neighbours:
                # calculate the node costs
                neighbour.heuristic_cost = self.calculate_heuristic_cost(neighbour, self.target)
                neighbour.turn_cost = self.calculate_turn_cost(neighbour)
                # if the came from at this location is empty, the current path is guaranteed to be the most optimal
                # path found so far
                came_from_node = self.came_from[int(neighbour.location[0])][int(neighbour.location[1])]
                if (came_from_node is None) or (neighbour.travelled_cost < came_from_node.travelled_cost):
                    came_from_node = current_node
                    came_from_node.travelled_cost = neighbour.travelled_cost
                    self.came_from[int(neighbour.location[0])][int(neighbour.location[1])] = came_from_node
                    if neighbour not in open_set:
                        open_set.append(neighbour)
        return status

    @staticmethod
    def calculate_heuristic_cost(node: Node, target: Node) -> float:
        """
        calculate the routing heuristic value for a node
        :param node: the node to calculate the heuristic for
        :param target: the target routing node
        :return: None
        """
        x_distance = abs(target.location[0] - node.location[0])
        y_distance = abs(target.location[1] - node.location[1])
        return x_distance + y_distance

    def calculate_turn_cost(self, node: Node) -> float:
        """
        Calculate the transition penalty for a route candidate
        :param node: The node to route to
        :return: turn penalty
        """
        parent = node.parent
        current_node = node
        turns = 0

        if current_node.location[1] == parent.location[1]:
            direction = AgentCoordinates.X
        else:
            direction = AgentCoordinates.Y

        while parent is not None:
            if current_node.location[1] == parent.location[1]:
                new_direction = AgentCoordinates.X
            else:
                new_direction = AgentCoordinates.Y
            if new_direction != direction:
                direction = new_direction
                turns += 1
            current_node = parent
            parent = parent.parent
        return turns * self._turn_factor

    def construct_node_path(self, target_node: Node) -> list:
        """
        Reconstruct the routing path from target to start by connecting the nodes that the algorithm found as
        part of the path
        :return:
        """
        nodes = list()
        # append the target_node node to the node path
        nodes.append(target_node)
        previous_node = self.came_from[self.target.location[0]][self.target.location[1]]
        if previous_node is not None:
            while previous_node is not self.start:
                nodes.append(previous_node)
                previous_node = self.came_from[int(previous_node.location[0])][int(previous_node.location[1])]
        # append the start node
        nodes.append(self.start)
        # reverse the list so that it goes start to end
        nodes.reverse()
        return nodes

    def generate_new_nodes(self, parent: Node) -> list:
        """
        Creates a list of new nodes that have a parent node. Note: each of these nodes has a
        travel cost equal to the parent node plus one
        :param parent: The parent node
        :return: list of new nodes
        """
        new_nodes = list()
        # get the neighbour tiles from the arena object
        neighbours = self.arena.get_neighbours(parent.location[0], parent.location[1])
        agent_locations = [(agent.location.X, agent.location.Y) for agent in self.agents]
        for neighbour in neighbours:
            # create a node
            new_node = Node(neighbour, parent=parent)
            new_node.travelled_cost = parent.travelled_cost + 1
            # if the tile is free, add it to the set
            tile_state = self.arena.get_tile_state(neighbour[0], neighbour[1])
            if tile_state == TileState.FREE and neighbour not in agent_locations:
                new_nodes.append(new_node)
        return new_nodes

    def initialize_nodes(self, agent: Agent, target: tuple) -> None:
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
        # check for other agents at that location
        agent_locations = [(agent.location.X, agent.location.Y) for agent in self.agents]
        if (x, y) in agent_locations:
            return RoutingStatus.TARGET_RESERVED

        tile_state = self.arena.get_tile_state(x, y)
        if tile_state == TileState.BLOCKED:
            return RoutingStatus.TARGET_BLOCKED
        elif tile_state == TileState.RESERVED:
            return RoutingStatus.TARGET_RESERVED
        else:
            return RoutingStatus.SUCCESS

