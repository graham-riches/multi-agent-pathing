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
from core.arena import Arena
from core.agent import Agent, AgentCoordinates
from routing.routing_algorithm import SingleAgentAlgorithm, Node
from routing.status import RoutingStatus
from routing.biased_grid import BiasedGrid, BiasedDirection
from core.tile import TileState


class AStarNode(Node):
    def __init__(self, location: tuple, parent=None) -> None:
        """
        Generate a node at a specific location.
        :param location: the location tuple (x,y)
        :param parent: A parent Node
        """
        # initialize parent node class
        super(AStarNode, self).__init__(location, parent)

        # store the value of any algorithm weights. Note: the travelled cost is directly related to the parent node
        self.total_cost = 0
        self._travelled_cost = 0  # penalize distance travelled
        self._heuristic_cost = 0  # penalize being far from the goal
        self._turn_cost = 0  # penalize turning or changing directions
        self._inline_cost = 0  # penalize being inline with another agent
        self._biased_cost = 0  # penalize routing through a square with a preferred direction

    @property
    def travelled_cost(self) -> float:
        """
        the cost associated with the total distance travelled
        :return:
        """
        return self._travelled_cost

    @travelled_cost.setter
    def travelled_cost(self, cost: float) -> None:
        self._travelled_cost = cost
        self.calculate_total_cost()

    @property
    def heuristic_cost(self) -> float:
        """
        the heuristic cost, which is roughly the geometric distance to the goal
        :return:
        """
        return self._heuristic_cost

    @heuristic_cost.setter
    def heuristic_cost(self, cost: float) -> None:
        self._heuristic_cost = cost
        self.calculate_total_cost()

    @property
    def turn_cost(self) -> float:
        """
        the cost associated with making a turn
        :return:
        """
        return self._turn_cost

    @turn_cost.setter
    def turn_cost(self, cost: float) -> None:
        self._turn_cost = cost
        self.calculate_total_cost()

    @property
    def inline_cost(self) -> float:
        """
        the cost associated with being inline with another agent
        :return:
        """
        return self._inline_cost

    @inline_cost.setter
    def inline_cost(self, cost: float) -> None:
        self._inline_cost = cost
        self.calculate_total_cost()

    @property
    def biased_cost(self) -> float:
        """
        the cost associated with routing through a square with a preferred direction
        :return:
        """
        return self._biased_cost

    @biased_cost.setter
    def biased_cost(self, cost: float) -> None:
        self._biased_cost = cost
        self.calculate_total_cost()

    def calculate_total_cost(self) -> None:
        """
        Calculate the total routing cost for a node
        :return:
        """
        self.total_cost = self._heuristic_cost + self._travelled_cost + \
                          self._turn_cost + self._inline_cost + self._biased_cost

    def __eq__(self, other) -> bool:
        """
        compare two Node objects
        :param other: another node
        """
        return self.location == other.location


class AStar(SingleAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list, biased_grid: BiasedGrid) -> None:
        """
        Initialize an A* search object given a particular arena and a list of agents. Agent locations are considered
        as blockages that must be handled in the search
        """
        # initialize parent class
        super(AStar, self).__init__(arena, agents, biased_grid)
        # initialize routing components
        self.start = None
        self.target = None
        self.came_from = None
        self._turn_factor = 0
        self._inline_factor = 0
        self._bias_factor = 0
        self.reset()

    @property
    def turn_factor(self) -> float:
        """
        get the penalty cost associated with making a turn
        :return:
        """
        return self._turn_factor

    @turn_factor.setter
    def turn_factor(self, factor: float) -> None:
        self._turn_factor = factor

    @property
    def inline_factor(self) -> float:
        """
        Get the penalty cost associated with being in-line with another agent
        :return:
        """
        return self._inline_factor

    @inline_factor.setter
    def inline_factor(self, factor: float) -> None:
        self._inline_factor = factor

    @property
    def bias_factor(self) -> float:
        """
        get the penalty associated with moving across a square against its preferred direction
        :return:
        """
        return self._bias_factor

    @bias_factor.setter
    def bias_factor(self, bias: float) -> None:
        self._bias_factor = bias

    def reset(self) -> None:
        """
        Reset the algorithms internal properties
        :return:
        """
        # re-initialize the algorithm parent class
        super(AStar, self).__init__(self.arena, self.agents, self.biased_grid)
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
                if self._turn_factor:
                    neighbour.turn_cost = self.calculate_turn_cost(neighbour)
                if self._inline_factor:
                    neighbour.inline_cost = self.calculate_inline_cost(neighbour)
                if self._bias_factor:
                    neighbour.biased_cost = self.calculate_biased_cost(neighbour)
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
    def calculate_heuristic_cost(node: AStarNode, target: AStarNode) -> float:
        """
        calculate the routing heuristic value for a node
        :param node: the node to calculate the heuristic for
        :param target: the target routing node
        :return: None
        """
        x_distance = abs(target.location[0] - node.location[0])
        y_distance = abs(target.location[1] - node.location[1])
        return x_distance + y_distance

    def calculate_turn_cost(self, node: AStarNode) -> float:
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

    def calculate_inline_cost(self, node: AStarNode) -> float:
        """
        calculate the cost of being inline with another agent
        :param node:
        :return:
        """
        x_size, y_size = self.arena.get_dimensions()
        # get all tiles inline with
        y_tiles = [(node.location[0], y) for y in range(y_size)]
        x_tiles = [(x, node.location[1]) for x in range(x_size)]
        tiles = y_tiles
        tiles.extend(x_tiles)
        for tile in tiles:
            state = self.arena.get_tile_state(tile[0], tile[1])
            if state == TileState.AGENT_TARGET:
                return self._inline_factor
        return 0

    def calculate_biased_cost(self, node: AStarNode) -> float:
        """
        calculate the cost of routing against the preferred direction of a square
        :param node: the node to calculate the cost of
        :return: cost
        """
        bias = self.biased_grid[node.location[0], node.location[1]]
        cost = 0
        if bias != BiasedDirection.BIAS_NONE:
            parent = node.parent
            if node.location[0] > parent.location[1]:
                cost = self._bias_factor if bias != BiasedDirection.BIAS_X_POSITIVE else 0
            elif node.location[0] < parent.location[1]:
                cost = self._bias_factor if bias != BiasedDirection.BIAS_X_NEGATIVE else 0
            elif node.location[1] > parent.location[1]:
                cost = self._bias_factor if bias != BiasedDirection.BIAS_Y_POSITIVE else 0
            elif node.location[1] < parent.location[1]:
                cost = self._bias_factor if bias != BiasedDirection.BIAS_Y_NEGATIVE else 0
        return cost

    def construct_node_path(self, target_node: AStarNode) -> list:
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

    def generate_new_nodes(self, parent: AStarNode) -> list:
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
            new_node = AStarNode(neighbour, parent=parent)
            new_node.travelled_cost = parent.travelled_cost + 1
            # if the tile is free and routing is allowed in that direction, add it to the list
            route_direction_valid = self.is_direction_valid(parent.location, neighbour)
            tile_state = self.arena.get_tile_state(neighbour[0], neighbour[1])
            if tile_state == TileState.FREE and route_direction_valid and neighbour not in agent_locations:
                new_nodes.append(new_node)
        return new_nodes

    def initialize_nodes(self, agent: Agent, target: tuple) -> None:
        """
        generate the starting and ending nodes
        :param agent: the agent
        :param target: the target location tuple (x,y)
        :return: None
        """
        self.start = AStarNode((agent.location.X, agent.location.Y))
        self.target = AStarNode(target)

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

    def is_direction_valid(self, parent: tuple, child: tuple) -> bool:
        """
        check if routing in a specific direction is allowed based on the biased grid
        :param parent: the parent location tuple
        :param child: the child location tuple
        :return: true if route is valid
        """
        bias = self.biased_grid[child]
        if bias < BiasedDirection.ONLY_X_POSITIVE:
            return True

        if child[0] == parent[0]:
            # routing in Y
            if bias == BiasedDirection.ONLY_X_POSITIVE or bias == BiasedDirection.ONLY_X_NEGATIVE:
                return False
            elif bias == BiasedDirection.ONLY_Y_NEGATIVE:
                return child[1] < parent[1]
            elif bias == BiasedDirection.ONLY_Y_POSITIVE:
                return child[1] > parent[1]
        else:
            # routing in X
            if bias == BiasedDirection.ONLY_Y_NEGATIVE or bias == BiasedDirection.ONLY_Y_POSITIVE:
                return False
            elif bias == BiasedDirection.ONLY_X_POSITIVE:
                return child[0] > parent[0]
            elif bias == BiasedDirection.ONLY_X_NEGATIVE:
                return child[0] < parent[0]
