"""
    @file common_path.py
    @brief GA simulation setup for the common_path simulation file
    @author Graham Riches
    @details
   This contains the simulation parameters and run-function setup for the GA to load
"""

import numpy as np
from core.benchmark import BenchmarkRunner
from routing.a_star import AStar
from routing.managers.sequential_rerouting import SequentialRerouting

# Simulation basic setup
BENCHMARK_FILEPATH = 'benchmarks/common_path.json'

# GA parameters setup
inline_penalty = (0, 5, False)
turn_penalty = (0, 5, False)
bias_penalty = (0, 100, False)
agent_max_squares = (1, 10, True)
grid_bias = [(0, 4, True) for i in range(90)]
parameters = [inline_penalty, turn_penalty, bias_penalty, agent_max_squares]
parameters.extend(grid_bias)


# GA miscellaneous setup
fitness_one_label = 'Cycles to Complete'
fitness_two_label = 'Squares Travelled'
save = True
filename = 'common_path'


def simulation(simulation_parameters: list, render: bool) -> tuple:
    """
    run the simulation
    :param simulation_parameters: all the parameters
    :param render: enable or disable rendering
    :return: fitness values
    """
    runner = BenchmarkRunner(BENCHMARK_FILEPATH)
    runner.load_configuration()
    runner.render = render

    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents, runner.biased_grid)
    a_star.inline_factor = simulation_parameters[0]
    a_star.turn_factor = simulation_parameters[1]
    a_star.bias_factor = simulation_parameters[2]
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    routing_manager.agent_max_distance = [simulation_parameters[3] for i in range(len(runner.agents))]
    runner.routing_manager = routing_manager
    # set the biased grid very hacky!
    count = 0
    for i in range(1, 10):
        for j in range(0, 10):
            runner.biased_grid[i, j] = simulation_parameters[4 + count]
            count += 1
    cycles, total_squares = runner.run()
    line = ''.join('-' for i in range(100))
    print(line)
    print('PARAMETERS: {}'.format(simulation_parameters))
    print('CYCLES: {}, SQUARES: {}'.format(cycles, total_squares))
    print('{}\n'.format(line))
    return cycles, total_squares


def run(simulation_parameters: list) -> tuple:
    """
    Function to pass into the GA for optimization
    :param simulation_parameters: list of all parameters
    :return: f1 and f2 fitness values
    """
    return simulation(simulation_parameters, False)


def recall(individual: int, generation: int) -> None:
    """
    recall an individual from the simulation and replay it's outcomes
    :param individual: the individual id
    :param generation: which generation
    :return: None
    """
    simulation_data = np.load('{}.npy'.format(filename))
    simulation_parameters = simulation_data[generation, individual, :]
    render = True
    simulation(simulation_parameters, True)
