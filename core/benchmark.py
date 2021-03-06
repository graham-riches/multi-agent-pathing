"""
    @file benchmark.py
    @brief contains benchmark running utilities for testing algorithms
    @author Graham Riches
    @details
        Loads a simulation benchmark file and runs the simulation for a given algorithm.
   
"""
import numpy as np
import random as rd
import sys
import json
from routing.routing_algorithm import SingleAgentAlgorithm, MultiAgentAlgorithm
from core.render_engine import Renderer
from routing.a_star import AStar
from routing.managers.sequential_rerouting import SequentialRerouting
from routing.biased_grid import BiasedGrid, BiasedDirection
from core.agent import *
from core.arena import Arena


class BenchmarkRunner:
    def __init__(self, filepath: str) -> None:
        """
        Initialize a benchmark test from a configuration file (Json)
        :param filepath: path to the benchmark configuration
        """
        # read the JSON configuration into a dictionary
        with open(filepath, 'r') as jsonfile:
            self.config = json.load(jsonfile)

        # Benchmark simulation properties
        self._routing_algorithm = None
        self._manager_algorithm = None
        self._on_fail_cycles = 25000
        self._on_fail_squares = 500
        self._event_squares = None
        self.time_step = None
        self.dpi = None
        self._render = False
        self.renderer = None
        self.biased_grid = None
        self.routing_manager = None
        self.agents = None
        self.arena = None
        self.tasks = None

    @property
    def algorithm(self) -> SingleAgentAlgorithm:
        return self._routing_algorithm

    @algorithm.setter
    def algorithm(self, algorithm: SingleAgentAlgorithm) -> None:
        self._routing_algorithm = algorithm

    @property
    def routing_manager(self) -> MultiAgentAlgorithm:
        return self._manager_algorithm

    @routing_manager.setter
    def routing_manager(self, manager: MultiAgentAlgorithm) -> None:
        self._manager_algorithm = manager

    @property
    def render(self) -> bool:
        return self._render

    @render.setter
    def render(self, enable: bool) -> None:
        self._render = enable
        if enable:
            self.renderer = Renderer(self.arena, self.agents, self.routing_manager,
                                     self.biased_grid, self.time_step, self.dpi)
            self.renderer.event_tiles = self._event_squares

    def parse_simulation_properties(self) -> None:
        """
        Parse the configuration dictionary to get the simulation properties
        :return:
        """
        sim_config = self.config['simulation_properties']
        self.time_step = sim_config['timestep']
        self.dpi = sim_config['grid_dpi']
        self._render = sim_config['render']

    def parse_agents(self) -> None:
        """
        Creates the internal list of agents based on the sim config
        :return:
        """
        if self.time_step is None:
            return
        agents = self.config['agents']
        sim_agents = list()
        for agent in agents:
            location = agent['location']
            new_agent = Agent(location[0], location[1], self.time_step)
            acceleration = agent['acceleration']
            deceleration = agent['deceleration']
            velocity = agent['velocity']
            new_agent.set_kinematic_parameters(acceleration, deceleration, velocity)
            sim_agents.append(new_agent)
        self.agents = sim_agents

    def parse_arena(self) -> None:
        """
        Creates the simulation arena
        :return:
        """
        arena = self.config['arena_properties']
        arena_size = arena['size']
        sim_arena = Arena(arena_size[0], arena_size[1])
        blockages = arena['blockages']
        for blockage in blockages:
            sim_arena.set_blockage([blockage[0]], [blockage[1]])
        self.arena = sim_arena
        self.biased_grid = BiasedGrid(self.arena.get_dimensions())
        biased_squares = arena['biases']
        self._event_squares = arena['event_squares']
        for bias in biased_squares:
            self.biased_grid[bias[0], bias[1]] = bias[2]

    def parse_tasks(self) -> None:
        """
        parse the agent tasks into a list
        :return:
        """
        self.tasks = self.config['tasks']

    def load_configuration(self) -> None:
        """
        Main loading method which will create the entire simulation configuration. This calls the
        smaller parsing methods that are purpose-built for specific properties.
        :return: None
        """
        self.parse_simulation_properties()
        self.parse_arena()
        self.parse_agents()
        self.parse_tasks()
        # create the renderer if required
        if self._render:
            self.renderer = Renderer(self.arena, self.agents, self.routing_manager,
                                     self.biased_grid, self.time_step, self.dpi)

    def render_simulation(self) -> None:
        """
        render the simulation if enabled
        :return:
        """
        if self.render:
            self.renderer.render_arena()
            for agent_id, agent in enumerate(self.agents):
                self.renderer.render_agent(agent_id)
            self.renderer.update()

    def run(self) -> tuple:
        """
        run the simulation and render it if requested.
        :return:  the number of cycles to complete
        """
        cycles = 0
        total_distance = 0
        if (self._manager_algorithm is None) and (self._routing_algorithm is None):
            return cycles, total_distance

        # set the goal locations in the multi-agent algorithm
        for task in self.tasks:
            if task['task_id'] == 'route':
                parameters = task['task_parameters']
                targets = parameters['location']
                for goal in targets:
                    # check for empty goals and randomly select from the available event squares
                    if goal[0] is None:
                        idx = rd.randint(0, len(self._event_squares) - 1)
                        goal = self._event_squares[idx]
                    new_goal = (goal[0], goal[1])
                    self._manager_algorithm.add_agent_goal(parameters['agent_id'], new_goal)

        # initialize the simulation and run until complete
        self._manager_algorithm.initialize()
        while not self._manager_algorithm.is_simulation_complete():
            self._manager_algorithm.run_time_step()
            self.render_simulation()
            cycles += 1
            # check for a grid lock and return if the simulation is stalled
            if self._manager_algorithm.is_locked():
                return self._on_fail_cycles, self._on_fail_squares

        # calculate the total simulation distance
        for agent in self.agents:
            total_distance += agent.get_squares_travelled()
        return cycles, total_distance


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python benchmark.py benchmark.json')
    config_file = sys.argv[1]
    # load the configuration
    runner = BenchmarkRunner(config_file)
    runner.load_configuration()
    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents, runner.biased_grid)
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    routing_manager.route_by_most_distant = False
    runner.routing_manager = routing_manager
    runner.render = True
    run_cycles, distance = runner.run()
    print('BENCHMARK: Simulation cycles {}'.format(run_cycles))
