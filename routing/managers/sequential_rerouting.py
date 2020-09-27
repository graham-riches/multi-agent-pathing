"""
    @file sequential_rerouting.py
    @brief route agents towards their goal and re-route as agents complete tasks
    @author Graham Riches
    @details
        Route agents towards their goal squares and re-route them whenever an agent completes a task.
"""

from routing.routing_algorithm import MultiAgentAlgorithm, SingleAgentAlgorithm
from routing.status import RoutingStatus
from core.arena import Arena
from core.agent import *


class SequentialRerouting(MultiAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list, algorithm: SingleAgentAlgorithm) -> None:
        """
        create an instance of the sequential routing algorithm
        :param arena: the arena object
        :param agents: list of agents
        :param algorithm: main a_star routing algorithm
        """
        super(SequentialRerouting, self).__init__(arena, agents, algorithm)
        self.active_agents = [False for agent in self.agents]
        self.initialized = [False for agent in self.agents]

    def run_time_step(self) -> None:
        """
        main algorithm run function for a single time step. Attempt to route any inactive agents that do
        not have a path planned.
        :return: None
        """
        for idx, agent in enumerate(self.agents):
            agent.update()

            # initialize the agent if required
            if not self.initialized[idx]:
                self.initialize_agent(idx)

            # signal that the last task completed and route other agents
            if agent.state == AgentState.IDLE:
                if self.active_agents[idx]:
                    self.signal_agent_event(idx, AgentEvent.TASK_COMPLETED)
                    self.active_agents[idx] = False
                    self.route_on_completion()
                if not self.is_agent_at_goal(idx):
                    # check for any new tasks
                    self.start_new_task(idx)

    def initialize_agent(self, agent_id: int) -> None:
        """
        initialize an agent at the start of the sim
        :param agent_id: the agent id
        :return:
        """
        self.route(agent_id, self.agent_goals[agent_id])
        self.initialized[agent_id] = True

    def start_new_task(self, agent_id: int) -> None:
        """
        start a new agent task from it's queue
        :param agent_id: the agents id
        :return: None
        """
        if len(self.agent_tasks[agent_id]) > 0:
            new_task = self.agent_tasks[agent_id].pop(0)
            self.agents[agent_id].start_task(new_task)
            self.active_agents[agent_id] = True

    def route(self, agent_id: int, target: tuple) -> None:
        """
        route an agent to a target location
        :param agent_id: ID of the agent to rout
        :param target: target location tuple
        :return:
        """
        if (target is None) or (self.active_agents[agent_id]):
            return
        # clear the agents task queue and blockages if it is idle
        self.agent_tasks[agent_id] = list()
        while len(self.agent_reserved_squares[agent_id]) > 0:
            self.clear_last_task_blockage(agent_id)

        agent = self.agents[agent_id]
        # reset the main routing algorithm
        self.routing_algorithm.reset()
        status = self.routing_algorithm.route(agent, target)
        if status == RoutingStatus.SUCCESS:
            # create the path and queue the first routing task
            route_status = self.routing_algorithm.create_path()
            if route_status == RoutingStatus.SUCCESS:
                self.add_agent_task(agent_id, self.routing_algorithm.path[0])
                self.start_new_task(agent_id)

    def route_on_completion(self) -> None:
        """
        Attempt to route other agents when an agent task completed event is received
        :return:
        """
        for idx, agent in enumerate(self.agents):
            if (not self.is_agent_at_goal(idx)) and (not self.active_agents[idx]):
                self.route(idx, self.agent_goals[idx])



