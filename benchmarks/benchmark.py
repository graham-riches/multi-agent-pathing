"""
    @file benchmark.py
    @brief contains benchmark running utilities for testing algorithms
    @author Graham Riches
    @details
        Loads a simulation benchmark file and runs the simulation for a given algorithm.
   
"""
import json
import time
from routing.routing_algorithm import SingleAgentAlgorithm
from render_engine import Renderer
from routing.routing_manager import RoutingManager, AgentEvent
from routing.status import RoutingStatus
from routing.a_star import AStar
from agent import *
from arena import Arena


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
        self.time_step = None
        self.dpi = None
        self.render = False
        self.renderer = None
        self.algorithm = None
        self.routing_manager = None
        self.agents = None
        self.arena = None
        self.tasks = None

    def set_algorithm(self, algorithm: SingleAgentAlgorithm) -> None:
        """
        Set the pathing algorithm for the benchmark simulation. This is the final simulation dependency, so create
        the remaining objects that depend on this as well.
        :param algorithm: the algorithm object
        :return: None
        """
        self.algorithm = algorithm
        self.routing_manager = RoutingManager(self.arena, self.agents, self.algorithm)
        if self.render:
            self.renderer = Renderer(self.arena, self.agents, self.routing_manager, self.time_step, self.dpi)

    def parse_simulation_properties(self) -> None:
        """
        Parse the configuration dictionary to get the simulation properties
        :return:
        """
        sim_config = self.config['simulation_properties']
        self.time_step = sim_config['timestep']
        self.dpi = sim_config['grid_dpi']
        self.render = sim_config['render']

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

    def run_simulation(self) -> int:
        """
        run the simulation and render it if requested.
        :return:  the number of cycles to complete
        """
        cycles = 0
        while len(self.tasks) > 0:
            cycles += 1
            # attempt to start the tasks
            for idx, task in enumerate(self.tasks):
                if task['task_id'] == 'route':
                    parameters = task['task_parameters']
                    agent_id = parameters['agent_id']
                    location = parameters['location']
                    status = self.routing_manager.route(agent_id, location[0], location[1])
                    # pop the task if the route was successful, otherwise try again another cycle
                    if status == RoutingStatus.SUCCESS:
                        self.tasks.pop(idx)
            # update everything
            for agent_id, agent in enumerate(self.agents):
                state = agent.update()
                if state == AgentState.IDLE:
                    self.routing_manager.signal_agent_event(agent_id, AgentEvent.TASK_COMPLETED)
            # render if required
            self.render_simulation()
        return cycles


if __name__ == '__main__':
    # load the configuration
    runner = BenchmarkRunner('test_config.json')
    runner.load_configuration()
    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents)
    runner.set_algorithm(a_star)
    run_cycles = runner.run_simulation()
    print(run_cycles)

