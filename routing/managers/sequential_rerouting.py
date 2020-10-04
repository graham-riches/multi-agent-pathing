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
        self._route_by_most_distant = False  # default to being greedy

    @property
    def route_by_most_distant(self) -> bool:
        """
        route agents by farthest from goal first
        """
        return self._route_by_most_distant

    @route_by_most_distant.setter
    def route_by_most_distant(self, enable: bool) -> None:
        self._route_by_most_distant = enable

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
                    self.start_new_task(idx)

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
        self.agent_routing_state[agent_id] = status
        if status == RoutingStatus.SUCCESS:
            # create the path and queue the first routing task
            route_status = self.routing_algorithm.create_path()
            if route_status == RoutingStatus.SUCCESS:
                task = self.routing_algorithm.path[0]
                # check if the move task is less than the max length allowed and truncate if its too large
                if task.args[1] >= agent.max_distance:
                    task.args[1] = agent.max_distance
                self.add_agent_task(agent_id, task)
                self.start_new_task(agent_id)

    def get_agents_by_distance(self) -> list:
        """
        return a list of agents total X+Y distance to their goal sorted smallest
        to largest.
        :return: list of agents
        """
        distance_dict = dict()
        for agent_id, agent in enumerate(self.agents):
            distance_x = abs(self.agent_goals[agent_id][0] - agent.location.X)
            distance_y = abs(self.agent_goals[agent_id][1] - agent.location.Y)
            distance_dict[agent_id] = distance_x + distance_y
        # sort the dict and get the list of agents out of it
        sorted_by_distance = {dist: agent_id for dist, agent_id in sorted(distance_dict.items(), key=lambda item: item[1])}
        return [x for x, v in sorted_by_distance.items()]

    def route_on_completion(self) -> None:
        """
        Attempt to route other agents when an agent task completed event is received
        :return:
        """
        # check to route agents by farthest or closest.
        agents_by_distance = self.get_agents_by_distance()
        if self._route_by_most_distant:
            agents_by_distance.reverse()

        for agent_id in agents_by_distance:
            if (not self.is_agent_at_goal(agent_id)) and (not self.active_agents[agent_id]):
                self.route(agent_id, self.agent_goals[agent_id])
