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
from agent import Agent
from routing.routing_algorithm import Algorithm
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
        self.travelled_cost = 0 if parent is None else parent.travelled_cost + 1
        self.heuristic_cost = 0

    def calculate_heuristic(self, target) -> None:
        """
        calculate the routing heuristic value for a node
        :param target: the target routing node
        :return: None
        """
        x_distance = abs(target.location[0] - self.location[0])
        y_distance = abs(target.location[1] - self.location[1])
        self.heuristic_cost = x_distance**2 + y_distance**2

    def calculate_total_cost(self) -> None:
        """
        Calculate the total routing cost for a node
        :return:
        """
        self.total_cost = self.heuristic_cost + self.travelled_cost

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
        # initialize the algorithm
        super(AStar, self).__init__(arena, agents)
        # initialize routing components
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
            # TODO replace this with a sorted queue instead of two list operations
            node_scores = [node.total_cost for node in open_set]
            index = np.argmin(node_scores)
            current_node = open_set.pop(int(index))
            # check if the current node is equal to the target and return if true
            if current_node == self.target:
                status = RoutingStatus.SUCCESS
                self.construct_node_path()
                self.create_path()
                break

            neighbours = self.generate_new_nodes(current_node)
            for neighbour in neighbours:
                # calculate the node costs
                neighbour.calculate_heuristic(self.target)
                neighbour.calculate_total_cost()
                # if the came from at this location is empty, the current path is guaranteed to be the most optimal
                # path found so far
                came_from_node = self.came_from[neighbour.location[0]][neighbour.location[1]]
                if (came_from_node is None) or (neighbour.travelled_cost < came_from_node.travelled_cost):
                    came_from_node = current_node
                    came_from_node.travelled_cost = neighbour.travelled_cost
                    self.came_from[neighbour.location[0]][neighbour.location[1]] = came_from_node
                    if neighbour not in open_set:
                        open_set.append(neighbour)
        return status

    def construct_node_path(self) -> None:
        """
        Reconstruct the routing path from target to start by connecting the nodes that the algorithm found as
        part of the path
        :return:
        """
        # append the target node to the node path
        self.node_path.append(self.target)
        previous_node = self.came_from[self.target.location[0]][self.target.location[1]]
        while previous_node is not self.start:
            self.node_path.append(previous_node)
            previous_node = self.came_from[previous_node.location[0]][previous_node.location[1]]
        # reverse the list so that it goes start to end
        self.node_path.reverse()

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
        for neighbour in neighbours:
            # if the tile is free, add it to the set
            if self.arena.get_tile_state(neighbour[0], neighbour[1]) == TileState.FREE:
                new_nodes.append(Node(neighbour, parent=parent))
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
        tile_state = self.arena.get_tile_state(x, y)
        if tile_state == TileState.BLOCKED:
            return RoutingStatus.TARGET_BLOCKED
        elif tile_state == TileState.RESERVED:
            return RoutingStatus.TARGET_RESERVED
        else:
            return RoutingStatus.SUCCESS

