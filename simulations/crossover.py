"""
    @file crossover.py
    @brief GA simulation setup for the crossover benchmark
    @author Graham Riches
    @details
        GA simulation of the crossover benchmark scenario
"""

import numpy as np
from core.benchmark import BenchmarkRunner
from routing.a_star import AStar
from routing.managers.sequential_rerouting import SequentialRerouting

BENCHMARK_FILEPATH = 'benchmarks/crossover.json'

# GA parameters setup
inline_penalty = (0, 5, False)
turn_penalty = (0, 5, False)
agent_max_squares = [(1, 10, True) for i in range(10)]
parameters = [inline_penalty, turn_penalty]
parameters.extend(agent_max_squares)


# GA miscellaneous setup
fitness_one_label = 'Cycles to Complete'
fitness_two_label = 'Squares Travelled'
save = True
filename = 'crossover'


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
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    routing_manager.agent_max_distance = [simulation_parameters[2 + i] for i in range(len(runner.agents))]
    runner.routing_manager = routing_manager
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
