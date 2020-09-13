"""
    @file routing_manager.py
    @brief main routing manager for pathing
    @author Graham Riches
    @details
    manager object to control routing for an agent by managing what the arena/available squares
"""
from enum import Enum
from arena import Arena
from agent import Agent, AgentCoordinates, AgentTask
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
        # dictionary of functions to use as event handlers
        self.agent_callbacks = {AgentEvent.TASK_COMPLETED: self.update_agent_task}

    def update_agent_task(self, agent_id: int) -> None:
        """
        An agent has finished it's last routing task, so update it
        :param agent_id: the ID of the agent
        :return:
        """
        current_agent = self.agents[agent_id]
        task_list = self.agent_tasks[agent_id]
        # agent tasks are a FIFO, so pop the first task off the list
        if len(task_list) > 0:
            task_list.pop(0)
        if len(task_list):
            current_agent.start_task(task_list[0])

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
