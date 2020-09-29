"""
    @file sequential.py
    @brief route agents towards their goal sequentially using a_star.
    @author Graham Riches
    @details
        This is intended to be a rather poor multi-agent routing algorithm that will route all agents
        towards their target goal. The main premise is as follows:
            - route the first agent to it's goal reserving all squares as required
            - attempt to route the other agents
            - if any other agents CAN be routed to their goal, route them as well
            - after every task completed event, see if any other agents can be routed
            - continue until all agents have reached their goal location
"""

from routing.routing_algorithm import MultiAgentAlgorithm, SingleAgentAlgorithm
from routing.status import RoutingStatus
from core.arena import Arena
from core.agent import *


class Sequential(MultiAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list, algorithm: SingleAgentAlgorithm) -> None:
        """
        create an instance of the sequential routing algorithm
        :param arena: the arena object
        :param agents: list of agents
        :param algorithm: main a_star routing algorithm
        """
        super(Sequential, self).__init__(arena, agents, algorithm)

    def run_time_step(self) -> None:
        """
        main algorithm run function for a single time step. Attempt to route any inactive agents that do
        not have a path planned.
        :return: None
        """
        for idx, agent in enumerate(self.agents):
            agent.update()

            # signal that the last task completed and route other agents
            if agent.state == AgentState.IDLE:
                if self.active_agents[idx]:
                    self.signal_agent_event(idx, AgentEvent.TASK_COMPLETED)
                    self.route_on_completion()

                # check for new tasks in the agents task queue
                if not self.is_agent_at_goal(idx):
                    # check for any new tasks
                    self.start_new_task(idx)

    def route(self, agent_id: int, target: tuple) -> None:
        """
        route an agent to a target location
        :param agent_id: ID of the agent to rout
        :param target: target location tuple
        :return:
        """
        if target is None:
            return
        agent = self.agents[agent_id]
        # reset the main routing algorithm
        self.routing_algorithm.reset()
        status = self.routing_algorithm.route(agent, target)
        if status == RoutingStatus.SUCCESS:
            # create the path and queue the tasks
            route_status = self.routing_algorithm.create_path()
            if route_status == RoutingStatus.SUCCESS:
                for task in self.routing_algorithm.path:
                    self.add_agent_task(agent_id, task)

    def route_on_completion(self) -> None:
        """
        Attempt to route other agents when an agent task completed event is received
        :return:
        """
        for idx, agent in enumerate(self.agents):
            if not self.is_agent_at_goal(idx):
                self.route(idx, self.agent_goals[idx])
