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


class Node(ABC):
    @abstractmethod
    def __init__(self, location: tuple, parent=None) -> None:
        """
        Initialize a single routing node object
        :param location: a (X, Y) location tuple for the node
        :param parent: another Node object that is the parent of the current nodes for pathing
        """
        self.location = location
        self.parent = parent


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
    def __init__(self, arena: Arena, agents: list, algorithm: SingleAgentAlgorithm) -> None:
        """
        Creates a multi-agent routing algorithm. This manages routing a group of agents towards a goal.
        :param arena: arena object
        :param agents: lists of agents
        :param algorithm: the single agent routing algorithm to use
        """
        self.arena = arena  # the arena object
        self.agents = agents  # list of agents
        self.routing_algorithm = algorithm
        self.agent_tasks = [list() for agent in self.agents]  # empty task list for each agent
        self.agent_reserved_squares = [list() for agent in self.agents]  # empty reserved squares lists
        self.agent_goals = [None for agent in self.agents]  # goal location for each agent
        self.agent_callbacks = {AgentEvent.TASK_COMPLETED: self.clear_last_task_blockage}

    @abstractmethod
    def run_time_step(self) -> None:
        """
        abstract method to run a simulation time step. This method will contain the multi-agent management algorithm
        :return: None
        """
        pass

    @abstractmethod
    def route(self, agent_id: int, x_location: int, y_location: int) -> None:
        """
        Run the routing algorithm to route an agent to a specific location
        :param agent_id: the agent id
        :param x_location:
        :param y_location:
        :return: None
        """
        pass

    def is_simulation_complete(self) -> bool:
        """
        Returns true if all agents have successfully reached their target locations
        :return: boolean
        """
        for idx, agent in enumerate(self.agents):
            location = (agent.location.X, agent.location.Y)
            if location != self.agent_goals[idx]:
                return False
        return True

    def signal_agent_event(self, agent_id: int, event: AgentEvent) -> None:
        """
        signal to the routing manager that something of interest has happened
        :param agent_id: the ID of the agent that is signalling
        :param event: the event type
        :return: None
        """
        # call the callback associated with the event type
        self.agent_callbacks[event](agent_id)

    def clear_last_task_blockage(self, agent_id: int) -> None:
        """
        callback to call when an agent task has completed. This will clear
        the routing blocks from the last task
        :param agent_id: the id of the agent
        :return:
        """
        # clear any previous routing blockages
        reserved_squares = self.agent_reserved_squares[agent_id]
        if len(reserved_squares) > 0:
            squares = reserved_squares.pop()
            self.arena.clear_blockage(squares['x'], squares['y'])

    def add_agent_task(self, agent_id: int, task: AgentTask) -> None:
        """
        add a new task to an agents task list
        :param agent_id: the Id of the agent to append the task to
        :param task: the AgentTask object
        :return: None
        """
        # add routing blockages for move tasks
        if task.task_id == AgentTasks.MOVE:
            self.reserve_squares_for_routing(agent_id, task)
        self.agent_tasks[agent_id].append(task)

    def set_agent_goal(self, agent_id: int, location: tuple) -> None:
        """
        set the goal location for an agent. The algorithm will continually route to here until 'Done'
        :param agent_id: the id of the agent
        :param location: the target/goal location
        :return: None
        """
        self.agent_goals[agent_id] = location

    def reserve_squares_for_routing(self, agent_id: int, task: AgentTask) -> tuple:
        """
        Reserve grid squares for routing an agent. Note: if the agents task list depth is greater than 0, the
        reserved squares will start from the endpoint of the last task in the task list.
        :param agent_id: the agent id of the agent being routed
        :param task: the task containing the route details
        :return:
        """
        agent = self.agents[agent_id]
        x = int(agent.location.X)
        y = int(agent.location.Y)
        # calculate the routing squares based on the queued tasks
        for queued_task in self.agent_tasks[agent_id]:
            if queued_task.task_id == AgentTasks.MOVE:
                direction = queued_task.args[0]
                distance = queued_task.args[1]
                if direction == AgentCoordinates.X:
                    x += distance
                else:
                    y += distance

        task_args = task.args
        sign = np.sign(task_args[1])
        if task_args[0] == AgentCoordinates.X:
            x_start = x + 1 if sign > 0 else x - 1
            x_target = int(x_start + task_args[1])
            tiles = list(range(int(x_start), int(x_target), int(sign)))
            x_tiles = tiles
            y_tiles = [y]
        else:
            y_start = y + 1 if sign > 0 else y - 1
            y_target = int(y_start + task_args[1])
            tiles = list(range(int(y_start), int(y_target), int(sign)))
            x_tiles = [x]
            y_tiles = tiles
        self.arena.set_reserved(x_tiles, y_tiles)
        self.agent_reserved_squares[agent_id].append({'x': x_tiles, 'y': y_tiles})
        return x_tiles, y_tiles
