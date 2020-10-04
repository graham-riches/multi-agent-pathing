"""
    @file run_ga_optimization.py
    @brief run the genetic algorithm to optimize the routing simulation
    @author Graham Riches
    @details
        This packs the benchmarking simulation into a function that can be called by
        the genetic algorithm to calculate the fitness ranking of the simulation
"""

from core.benchmark import BenchmarkRunner
from optimization.genetic_algorithm import MultiObjectiveGeneticAlgorithm
from routing.a_star import AStar
from routing.managers.sequential_rerouting import SequentialRerouting


def run_simulation(parameters: list) -> tuple:
    runner = BenchmarkRunner('benchmarks/crossover.json')
    runner.load_configuration()
    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents)
    a_star.inline_factor = parameters[0]
    a_star.turn_factor = parameters[1]
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    agent_max_distance = parameters[2:]
    routing_manager.agent_max_distance = agent_max_distance
    runner.routing_manager = routing_manager
    cycles, total_squares = runner.run()
    line = ''.join('-' for i in range(100))
    print(line)
    print('PARAMETERS: {}'.format(parameters))
    print('CYCLES: {}, SQUARES: {}'.format(cycles, total_squares))
    print('{}\n'.format(line))
    return cycles, total_squares


if __name__ == '__main__':
    # load the configuration
    inline = (0, 5, False)
    turn_factor = (0, 5, False)
    # TODO don't hardcode the 10 and read it from the benchmark??
    agent_max_squares = [(1, 10, True) for i in range(10)]
    parameters = [inline, turn_factor]
    parameters.extend(agent_max_squares)
    population = 40
    generations = 50
    ga = MultiObjectiveGeneticAlgorithm(parameters, population, generations, run_simulation)
    ga.fitness_one_label = 'Cycles to Complete'
    ga.fitness_two_label = 'Squares Travelled'
    ga.run()
