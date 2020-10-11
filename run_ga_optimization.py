"""
    @file run_ga_optimization.py
    @brief run the genetic algorithm to optimize the routing simulation
    @author Graham Riches
    @details
        Calls a specific GA simulation setup script and runs the sim. CLI options
        for recalling a rendering of a specific trial
"""
import importlib
import argparse
from optimization.genetic_algorithm import MultiObjectiveGeneticAlgorithm


HELP_STRING = """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-Agent Pathing Genetic Algorithm Simulation Engine CLI

Author: Graham Riches
Date: October 11, 2020

Details:
    Run a benchmark multi-agent routing simulation with the multi-objective
    genetic algorithm. This CLI also provides options for recalling a specific
    individual to see a replay of an interesting solution!
    
Current Benchmarks Available:
    crossover - 10x10 crossover scenario
    common_path - routing through a common area

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=HELP_STRING, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('mode', metavar='Mode', help='choose the mode: simulate, recall')
    parser.add_argument('benchmark', metavar='Benchmark', help='benchmark simulation module name')
    parser.add_argument('-g', metavar='--generations', help='number of generations', default=50)
    parser.add_argument('-p', metavar='--population', help='population size', default=40)
    parser.add_argument('-i', metavar='--individual', help='individual number to recall for display', default=0)
    parser.add_argument('-n', metavar='--recall_generation', help='generation number to recall for display', default=0)
    args = parser.parse_args()
    args_dict = vars(args)

    mode = args_dict['mode']
    module = args_dict['benchmark']
    population_size = int(args_dict['p'])
    generations = int(args_dict['g'])
    recall_individual = int(args_dict['i']) - 1
    recall_generation = int(args_dict['n']) - 1

    # load the simulation engine
    simulation = importlib.import_module('simulations.{}'.format(module))

    if mode == 'simulate':
        ga = MultiObjectiveGeneticAlgorithm(simulation.parameters, population_size, generations, simulation.run)
        ga.fitness_one_label = simulation.fitness_one_label
        ga.fitness_two_label = simulation.fitness_two_label
        ga.save = simulation.save
        ga.filename = simulation.filename
        ga.run()
    elif mode == 'recall':
        simulation.recall(recall_individual, recall_generation)
