"""
    @file run_ga_optimization.py
    @brief run the genetic algorithm to optimize the routing simulation
    @author Graham Riches
    @details
        This packs the benchmarking simulation into a function that can be called by
        the genetic algorithm to calculate the fitness ranking of the simulation
"""

from core.benchmark import BenchmarkRunner
from optimization.genetic_algorithm import MultiObjectiveGeneticAlgorithm, Parameter
from routing.a_star import AStar
from routing.managers.sequential_rerouting import SequentialRerouting


def run_simulation(parameters: list) -> tuple:
    runner = BenchmarkRunner('benchmarks/crossover.json')
    runner.load_configuration()
    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents)
    a_star.inline_factor = parameters[0]
    a_star.turn_factor = parameters[1]
    for idx, agent in enumerate(runner.agents):
        agent.max_distance = parameters[2 + idx]
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    runner.routing_manager = routing_manager
    cycles, total_squares = runner.run()
    # print('PARAMETERS: {}'.format(parameters))
    print('CYCLES: {}, SQUARES: {}'.format(cycles, total_squares))
    return cycles, total_squares


if __name__ == '__main__':
    # load the configuration
    inline = Parameter(0, 5, False)
    turn_factor = Parameter(0, 5, False)
    # TODO don't hardcode the 10 and read it from the benchmark
    agent_max_squares = [Parameter(1, 10, True) for i in range(10)]
    parameters = [inline, turn_factor]
    parameters.extend(agent_max_squares)
    population = 20
    generations = 30
    ga = MultiObjectiveGeneticAlgorithm(parameters, population, generations, run_simulation)
    ga.run()
