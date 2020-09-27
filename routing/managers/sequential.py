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
from arena import Arena
from agent import *


class Sequential(MultiAgentAlgorithm):
    def __init__(self, arena: Arena, agents: list, algorithm: SingleAgentAlgorithm) -> None:
        """
        create an instance of the sequential routing algorithm
        :param arena: the arena object
        :param agents: list of agents
        :param algorithm: main a_star routing algorithm
        """
        super(Sequential, self).__init__(arena, agents, algorithm)
        self.active_agents = [False for agent in self.agents]

    def run_time_step(self) -> None:
        """
        main algorithm run function for a single time step. Attempt to route any inactive agents that do
        not have a path planned.
        :return: None
        """
        for idx, agent in enumerate(self.agents):
            if (agent.state == AgentState.IDLE) and (not self.is_agent_at_goal(idx)):
                # signal that the last task completed
                if self.active_agents[idx]:
                    self.signal_agent_event(idx, AgentEvent.TASK_COMPLETED)
                    self.active_agents[idx] = False
                # check for any new tasks
                if len(self.agent_tasks[idx]) > 0:
                    new_task = self.agent_tasks[idx].pop(0)
                    self.agents[idx].start_task(new_task)
                    self.active_agents[idx] = True
                else:
                    # try to route the agent
                    self.route(idx, self.agent_goals[idx])

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


