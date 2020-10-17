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
from core.tile import TileState


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
        self._detect_stalled_simulation_cycles = 15  # consecutive simulation cycles without movement default
        self._stall_detect_count = 0

    @property
    def stall_detect_cycles(self) -> int:
        return int(self._detect_stalled_simulation_cycles)

    @stall_detect_cycles.setter
    def stall_detect_cycles(self, cycles: int) -> None:
        self._detect_stalled_simulation_cycles = int(cycles)

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

                # check for new goals if the current goal is completed
                if self.is_agent_at_goal(idx):
                    if not self.agent_goals_completed(idx):
                        self.update_agent_goal(idx)
                        self.route_on_completion()

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

        # check that the target location is free, and if not, route to a nearby square
        tile_state = self.arena.get_tile_state(target[0], target[1])
        agent_locations = [(agent.location.X, agent.location.Y) for agent in self.agents]
        if tile_state == TileState.RESERVED or tile_state == TileState.AGENT_TARGET or target in agent_locations:
            neighbours = self.arena.get_neighbours(target[0], target[1])
            for neighbour in neighbours:
                if self.arena.get_tile_state(neighbour[0], neighbour[1]) == TileState.FREE:
                    target = neighbour
                    break

        agent = self.agents[agent_id]
        # reset the main routing algorithm and route the agent
        self.routing_algorithm.reset()
        status = self.routing_algorithm.route(agent, target)
        self.agent_routing_state[agent_id] = status

        if status == RoutingStatus.SUCCESS:
            # create the path and queue the first routing task
            route_status = self.routing_algorithm.create_path()
            if route_status == RoutingStatus.SUCCESS:
                task = self.routing_algorithm.path[0]
                # check if the move task is less than the max length allowed and truncate if its too large
                direction = np.sign(task.args[1])
                if abs(task.args[1]) >= self.agent_max_distance[agent_id]:
                    task.args[1] = self.agent_max_distance[agent_id] * direction
                self.add_agent_task(agent_id, task)
                self.start_new_task(agent_id)

    def is_locked(self) -> bool:
        """
        Test if all agents have become locked in the simulation
        :return: boolean true if simulation is frozen/locked up
        """
        blocked = True
        for idx, agent in enumerate(self.agents):
            if agent.state != AgentState.IDLE:
                return False
        self.route_on_completion()
        self._stall_detect_count = self._stall_detect_count + 1 if blocked else 0
        if self._stall_detect_count >= self._detect_stalled_simulation_cycles:
            return True
        else:
            return False

    def get_agents_by_distance(self) -> list:
        """
        return a list of agents total X+Y distance to their goal sorted smallest
        to largest.
        :return: list of agents
        """
        distance_dict = dict()
        for agent_id, agent in enumerate(self.agents):
            agent_current_goal = self.agent_goals[agent_id][0]
            distance_x = abs(agent_current_goal[0] - agent.location.X)
            distance_y = abs(agent_current_goal[1] - agent.location.Y)
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
                agent_goal = self.agent_goals[agent_id][0]
                self.route(agent_id, agent_goal)
