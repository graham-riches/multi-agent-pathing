"""
    @file routing_manager.py
    @brief main routing manager for pathing
    @author Graham Riches
    @details
    manager object to control routing for an agent by managing what the arena/available squares
"""
from enum import Enum
import numpy as np
from arena import Arena
from agent import *
from tile import TileState


class AgentEvent(Enum):
    TASK_COMPLETED = 1


class RoutingManager:
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Start a routing manager object
        :param arena: the simulation arena
        :param agents: list of agents that need to be routed
        """
        self.arena = arena
        self.agents = agents
        self.agent_tasks = [list() for agent in self.agents]  # empty task list for each agent
        self.agent_reserved_squares = [list() for agent in self.agents]  # empty reserved squares list
        # dictionary of functions to use as event handlers
        self.agent_callbacks = {AgentEvent.TASK_COMPLETED: self.update_agent_task}

    def update_agent_task(self, agent_id: int) -> None:
        """
        An agent has finished it's last routing task, so update the task
        :param agent_id: the ID of the agent
        :return:
        """
        # clear any previous routing blockages
        reserved_squares = self.agent_reserved_squares[agent_id]
        if len(reserved_squares) > 0:
            squares = reserved_squares.pop()
            self.arena.clear_blockage(squares['x'], squares['y'])
        current_agent = self.agents[agent_id]
        task_list = self.agent_tasks[agent_id]
        if len(task_list) > 0:
            task = task_list.pop(0)
            if task.task_id == AgentTasks.MOVE:
                x, y = self.reserve_squares_for_routing(current_agent, task)
                self.agent_reserved_squares[agent_id].append({'x': x, 'y': y})
            current_agent.start_task(task)

    def reserve_squares_for_routing(self, agent: Agent, task: AgentTask) -> tuple:
        """
        Reserve grid squares for routing an agent
        :param agent: the agent being routed
        :param task: the task containing the route details
        :return:
        """
        x = int(agent.location.X)
        y = int(agent.location.Y)
        args = task.args
        sign = np.sign(args[1])
        if args[0] == AgentCoordinates.X:
            x_start = x + 1 if sign > 0 else x - 1
            x_target = int(x_start + args[1])
            tiles = list(range(x_start, x_target, sign))
            x_tiles = tiles
            y_tiles = [y]
        else:
            y_start = y + 1 if sign > 0 else y - 1
            y_target = int(y_start + args[1])
            tiles = list(range(y_start, y_target, sign))
            x_tiles = [x]
            y_tiles = tiles
        self.arena.set_reserved(x_tiles, y_tiles)
        return x_tiles, y_tiles

    def add_agent_task(self, agent_id: int, task: AgentTask) -> None:
        """
        add a new task to an agents task list
        :param agent_id: the Id of the agent to append the task to
        :param task: the AgentTask object
        :return: None
        """
        self.agent_tasks[agent_id].append(task)

    def signal_agent_event(self, agent_id: int, event: AgentEvent) -> None:
        """
        signal to the routing manager that something of interest has happened
        :param agent_id: the ID of the agent that is signalling
        :param event: the event type
        :return: None
        """
        # call the callback
        self.agent_callbacks[event](agent_id)
