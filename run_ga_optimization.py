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
    runner = BenchmarkRunner('benchmarks/common_path.json')
    runner.load_configuration()
    # create a new algorithm and attach it to the simulation
    a_star = AStar(runner.arena, runner.agents, runner.biased_grid)
    a_star.inline_factor = parameters[0]
    a_star.turn_factor = parameters[1]
    a_star.bias_factor = parameters[2]
    runner.algorithm = a_star
    routing_manager = SequentialRerouting(runner.arena, runner.agents, runner.algorithm)
    routing_manager.agent_max_distance = [parameters[3] for i in range(len(runner.agents))]
    runner.routing_manager = routing_manager
    # set the biased grid very hacky!
    count = 0
    for i in range(1, 10):
        for j in range(0, 10):
            runner.biased_grid[i, j] = parameters[4 + count]
            count += 1
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
    bias_factor = (0, 100, False)
    agent_max_squares = (1, 10, True)
    grid_bias = [(0, 4, True) for i in range(90)]
    parameters = [inline, turn_factor, bias_factor, agent_max_squares]
    parameters.extend(grid_bias)
    population = 40
    generations = 50
    ga = MultiObjectiveGeneticAlgorithm(parameters, population, generations, run_simulation)
    ga.fitness_one_label = 'Cycles to Complete'
    ga.fitness_two_label = 'Squares Travelled'
    ga.run()
