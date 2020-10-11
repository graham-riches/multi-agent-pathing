"""
@file genetic_algorithm.py
@brief multi-objective evolutionary algorithm
@author Conrad Bingham
@created August 29, 2016
@details
    Implementation of a modified NSGA-II (Non-dominated elitist Sorting Genetic
    Algorithm) applied to a real-parameter multivariable problem. The goal of this
    algorithm is to robustly find a consistent pareto-optimal front of solutions to
    a multi-objective optimisation problem.

    The following main operators are used:
    -Crowded Tournament Selection -> to create mating pool (uses derangement)
    -Explicit Prior Generation Elitism -> to ensure best solutions are not lost
    -N Point Real Parameter Blended Crossover -> to crossover non-binary solutions
    -Real Parameter Non-Uniform Mutation -> to randomly mutate solutions
    -Pareto Domination Ranking -> to rank individual fitness based on domination
    -Crowding Distance Ranking -> to ensure distribution on the pareto front

    Lets first go over the algorithm behind CNSGA, in simple form:

    Create random population P of size N
    Evaluate performance in both objectives for each member in P
    Evaluate pareto front rank F of each member in P
    Evaluate crowding value Ci of each member in their rank F
    For g=1:generations
        Using crowded tournament selection, create mating pool of elites
        Using mating ppol and blended crossover create child population of size N
            (Retain previous generations population!)
        Apply mutation to child population
        Evaluate performance in both objectives for each member in child population
            (This step involves a physical test only if the simulation deems safe)
        Add child population to parent population, yielding population P of size 2N
        Evaluate pareto front rank F of each member in P
        Evaluate crowding value Ci of each member in their rank F
        Sort population of size 2N first by F, then by Ci
        Remove lower half of population, creating group of size N
            (Retain F and Ci values for each)
    Repeat


    TO RUN:

    1) Have genetic_algorithm.py and MOEAFunc.py in the same folder
    2) Create a subfolder called 'PLOTS'
    3) Run this script
    4) In command line, call the MOEA function, ex: >>MOEA(20,20,1)
    5) Profit

-------------------------------------------------------------------------------------------
UPDATE: October 3, 2020
@author: Graham Riches
@details
    Started porting this bad boy over to work with the multi-agent pathing simulation :)

"""

from __future__ import division
import numpy as np
import os
import random as rd
import matplotlib.pylab as plt
from optimization.kursawe import get_kursawe_fitness


class MultiObjectiveGeneticAlgorithm:
    def __init__(self, parameters: list, population_size: int,
                 generations_to_run: int, fitness_function: callable) -> None:
        """
        Create an instance of the MOEA algorithm. NOTE: most of this algorithm is done using 2-dimensional arrays
        and has been optimized for speed, hence a few things may be somewhat hard to understand! The biggest one
        is that for each candidate in a population, the fitness, ranking, and crowding distances are tacked
        onto the end of the parameters list in the array. This makes it trivial-ish and relatively fast to sort the
        population by its fitness ranking and whatnot. So beware of that ¯\_(ツ)_/¯

        :param parameters: list of tuples. These are structured as (lower_limit, upper_limit, is_integer). NOTE:
                           parameters should have a length that is equal to a multiple of 2
        :param population_size: total size of the GA population to create
        :param generations_to_run: how many generations to run
        :param fitness_function: callable function to calculate the fitness. Input should be an array or list
                                 where the length is equal to the total number of parameters and should return a
                                 length 2 tuple containing (f1, f2)
        """
        # individuals and parameters
        self._total_parameters = len(parameters)
        self._individual_size = self._total_parameters + 4  # NOTE: this is to store [f1, f2, R, d] as part of 2D array
        self._parameter_upper_bounds = [parameter[1] for parameter in parameters]
        self._parameter_lower_bounds = [parameter[0] for parameter in parameters]
        self._parameter_is_integer = [parameter[2] for parameter in parameters]
        self._f1_index = self._total_parameters
        self._f2_index = self._total_parameters + 1
        self._rank_index = self._total_parameters + 2
        self._crowding_index = self._total_parameters + 3

        # simulation, generation, population etc.
        self._population_size = int(population_size) if population_size % 2 == 0 else int(population_size + 1)
        self._total_generations = int(generations_to_run) if generations_to_run >= 1 else 1
        self._current_generation = 1
        self._population = None
        self._population_history = None  # 3D array of populations
        self._fitness_bounds = {'min_f1': 0, 'max_f1': None, 'min_f2': None, 'max_f2': None}
        self._max_rank = None

        # modifiable properties
        self._fitness_function = fitness_function  # settable fitness function depending on the scenario
        self._mutation_multiplier = 1 / population_size  # default value is 1/n, can be set via property
        self._active_parameters = 0  # number of parameters we are actually evolving (Set to 0 for automatic)
        self._mutation_uniformity = 1  # Mutation non-uniformity, ~.5 is low, ~4 is high
        self._blended_crossover_probability = .5  # percent chance that a crossover will be blended between parents
        self._maximum_child_crossovers = 3  # maximum number of crossovers applied per child

        # debugging and data saving attributes
        self._save_data = False
        self._filename = 'test'
        self._plots_directory = 'PLOTS'  # create if it does not exist
        self._f1_plot_label = 'f1'
        self._f2_plot_label = 'f2'
        if not os.path.exists(self._plots_directory):
            os.mkdir(self._plots_directory)

    @property
    def save(self) -> bool:
        return self._save_data

    @save.setter
    def save(self, enable: bool) -> None:
        self._save_data = enable

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        self._filename = filename

    @property
    def plot_directory(self) -> str:
        return self._plots_directory

    @plot_directory.setter
    def plot_directory(self, directory: str) -> None:
        self._plots_directory = directory

    @property
    def fitness_one_label(self) -> str:
        return self._f1_plot_label

    @fitness_one_label.setter
    def fitness_one_label(self, label: str) -> None:
        self._f1_plot_label = label

    @property
    def fitness_two_label(self) -> str:
        return self._f2_plot_label

    @fitness_two_label.setter
    def fitness_two_label(self, label: str) -> None:
        self._f2_plot_label = label

    @property
    def mutation_multiplier(self) -> float:
        return self._mutation_multiplier

    @mutation_multiplier.setter
    def mutation_multiplier(self, rate: float) -> None:
        if rate < 0 or type(rate) is not float:
            print('WARNING: mutation multiplier should be a real floating point value greater than zero')
        if (1 / self._population_size) * rate >= 0.99:
            print('WARNING: mutation multiplier is high enough to guarantee mutations for every gene')
        self._mutation_multiplier = float(abs(rate))

    @property
    def active_parameters(self) -> int:
        return self._active_parameters

    @active_parameters.setter
    def active_parameters(self, active_parameters: int) -> None:
        self._active_parameters = int(active_parameters)

    @property
    def mutation_uniformity(self) -> float:
        return self._mutation_uniformity

    @mutation_uniformity.setter
    def mutation_uniformity(self, uniformity: float) -> None:
        self._mutation_uniformity = float(uniformity)

    @property
    def crossover_probability(self) -> float:
        return self._blended_crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, probability: float) -> None:
        self._blended_crossover_probability = float(probability)

    @property
    def max_crossovers(self) -> int:
        return self._maximum_child_crossovers

    @max_crossovers.setter
    def max_crossovers(self, limit: int) -> None:
        self._maximum_child_crossovers = int(limit)

    def create_population(self) -> np.ndarray:
        """
        Create the GA population
        :return: None
        """
        population = np.zeros(shape=(self._population_size, self._individual_size))
        self._population_history = np.zeros(shape=(self._total_generations,
                                                   self._population_size,
                                                   self._individual_size))
        # create a random population within the specified parameter bounds
        for individual in range(self._population_size):
            for parameter in range(self._total_parameters):
                parameter_range = self._parameter_upper_bounds[parameter] - self._parameter_lower_bounds[parameter]
                population[individual, parameter] = rd.random() * parameter_range + self._parameter_lower_bounds[parameter]
        # round if required
        for parameter in range(self._total_parameters):
            if self._parameter_is_integer[parameter]:
                population[:, parameter] = population[:, parameter].round()
        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate the current population by calculating the fitness of each individual and ranking them
        :param population: the population to rank
        :return: None
        """
        # run through all individuals and calculate their fitness
        for individual in range(self._population_size):
            for parameter in range(self._total_parameters):
                # cast integer parameters if required
                if self._parameter_is_integer[parameter]:
                    population[individual, parameter] = round(population[individual, parameter], 0)
            # log the individual
            print('GA Progress: Generation {}, Individual {}'.format(self._current_generation, individual + 1))
            # calculate fitness
            f1, f2 = self._fitness_function(population[individual, 0:self._total_parameters])
            # assign the fitness values to the 2D population array for the current individual
            population[individual, self._f1_index] = f1
            population[individual, self._f2_index] = f2
        # get the bounds for the current population
        self._fitness_bounds['min_f1'] = min(population[:, self._f1_index])
        self._fitness_bounds['max_f1'] = max(population[:, self._f1_index])
        self._fitness_bounds['min_f2'] = min(population[:, self._f2_index])
        self._fitness_bounds['max_f2'] = max(population[:, self._f2_index])
        return population

    def dominates(self, individual_1: list, individual_2: list) -> bool:
        """
        Check if one individual dominates another. Returns true if individual 1 is better than individual 2
        :param individual_1: the first individuals array (parameter, plus fitness, etc.)
        :param individual_2:the second individuals array (parameter, plus fitness, etc.)
        :return: boolean
        """
        a_fitness_1 = individual_1[self._f1_index]
        a_fitness_2 = individual_1[self._f2_index]
        b_fitness_1 = individual_2[self._f1_index]
        b_fitness_2 = individual_2[self._f2_index]

        dominates = False
        if a_fitness_1 == b_fitness_1 and a_fitness_2 < b_fitness_2:
            dominates = True
        elif a_fitness_1 < b_fitness_1 and a_fitness_2 == b_fitness_2:
            dominates = True
        elif a_fitness_1 < b_fitness_1 and a_fitness_2 < b_fitness_2:
            dominates = True
        return dominates

    def crowded_tournament(self, population: np.ndarray) -> np.ndarray:
        """
        We want every solution to participate in the tournament twice, paired with
        random other entity, without ever pairing with itself, with a perfectly
        uniform probability. This requires that a uniform derangement be performed
        on the indices list

        :param population: the population to run the tournament on
        :return: new np array of individuals
        """
        population_size = np.shape(population)[0]
        competition_list = self.derangement(population_size)

        # create an empty mating pool
        mating_pool = np.zeros(shape=np.shape(population))

        # compare each individual with it's opponent
        for i in range(population_size):
            mating_pool[i, :] = self.competition(population[i, :], population[competition_list[i], :])
        return mating_pool

    def blended_n_point_crossover(self, population: np.ndarray) -> np.ndarray:
        """
        Perform crossover on the population to create a new population
        :param population: the population array
        :return: new population array
        """
        genome_child = np.zeros(shape=np.shape(population))
        population_size = np.shape(population)[0]
        # Create each individual by selecting two random parents from the mating pool
        for individual in range(population_size):
            parent_1 = rd.randint(0, population_size - 1)
            check_for_no_incest = True
            while check_for_no_incest:
                parent_2 = rd.randint(0, population_size - 1)
                check_for_no_incest = parent_1 == parent_2

            # select the parents from the mating pool
            parents = np.zeros(shape=(2, self._individual_size))
            parents[0, :] = population[parent_1, :]
            parents[1, :] = population[parent_2, :]

            # get index locations for crossover
            crossover_indices = self.crossover_index()

            # go through each gene and select from correct parent
            parent_id = 0
            for parameter in range(self._total_parameters):
                # swap parent at each crossover
                if crossover_indices[parameter]:
                    parent_id = 1 if parent_id == 0 else 0
                if self._blended_crossover_probability > rd.random() and crossover_indices[parameter]:
                    # blending selected
                    blend_amount = rd.random()
                    genome_child[individual, parameter] = blend_amount * parents[0, parameter] + (1 - blend_amount) * parents[1, parameter]
                else:
                    # no blending
                    genome_child[individual, parameter] = parents[parent_id, parameter]
        return genome_child

    def crossover_index(self) -> list:
        """
        create array of ones and zeros, where a one is the index of a crossover
        array has to be random, and contain n crossover points
        :return:
        """
        crossover_points = np.random.rand(self._total_parameters)
        # extract maximum n values, where n is number of crossovers, (1-nCross)
        n = rd.randint(1, self._maximum_child_crossovers)
        ind = crossover_points.argsort()[-n:][::-1]
        crossover_points = crossover_points * 0
        crossover_points[ind] = 1
        return crossover_points

    def mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Chance to mutate any gene to create a new population
        :param population: the population to mutate
        :return: new population array
        """
        scale = self._total_parameters if self._active_parameters == 0 else self._active_parameters
        mutation_chance = (1.0 / scale) * self._mutation_multiplier
        population_size = np.shape(population)[0]
        for individual in range(population_size):
            for gene in range(self._total_parameters):
                if rd.random() < mutation_chance:
                    # scale mutation by generation and use non-uniform mutator (Michalewicz, 1992)
                    tau = rd.randint(0, 1)
                    generation_ratio = self._current_generation / self._total_generations
                    multiplier = (1 - rd.random() ** ((1 - generation_ratio) ** self._mutation_uniformity))
                    if tau:
                        # move towards upper limit
                        population[individual, gene] = population[individual, gene] + (self._parameter_upper_bounds[gene] - population[individual, gene]) * multiplier
                    else:
                        # move towards lower limit
                        population[individual, gene] = population[individual, gene] - (population[individual, gene] - self._parameter_lower_bounds[gene]) * multiplier

        return population

    def slice(self, population: np.ndarray) -> np.ndarray:
        """
        Takes an overgrown population and shrinks it down to self._population_size with the most optimal individuals
        :param population: the population
        :return: shrunk population
        """
        idx = np.lexsort((population[:, self._crowding_index] * -1, population[:, self._rank_index]))
        population = population[idx]
        new_population = population[0:self._population_size, :]
        self._max_rank = int(max(new_population[:, self._rank_index]))
        return new_population

    def pareto_rank(self, population: np.ndarray) -> np.ndarray:
        """
        Rank the population individuals onto a pareto front.
        NOTE: this algorithm is pretty hefty and complicated. Beware!! This is mostly copied over from the original
              implementation
        :param population: the population to rank
        :return: sorted population array by pareto rank
        """
        complete = False  # initialize the algorithm to not complete
        rank = 0  # set the initial ranking
        not_ranked = -1
        population[:, self._rank_index] = not_ranked  # default initialize all rankings for the current generation
        population_size = np.shape(population)[0]
        while not complete:
            rank += 1
            # find the first unranked individual
            i = 0
            found_unranked = False
            while not found_unranked:
                if population[i, self._rank_index] == not_ranked:
                    population[i, self._rank_index] = rank
                    found_unranked = True
                else:
                    i += 1

            # check all other unranked solutions against entire unranked or current ranked set
            for j in range(i + 1, population_size):
                # check to see if this individual ever gets dominated by another
                dominated = False
                for k in range(i, j):
                    # ensure genome has not yet been ranked and its comparator is unranked or in the current rank set
                    if (population[j, self._rank_index] == not_ranked and
                        (population[k, self._rank_index] == not_ranked or
                         population[k, self._rank_index] == rank)):
                        # if current individual dominates comparator, set comparators rank to -1
                        if self.dominates(population[j, :], population[k, :]):
                            population[k, self._rank_index] = not_ranked
                        # if current individual is dominated by comparator, it does not belong in this rank set
                        if self.dominates(population[k, :], population[j, :]):
                            dominated = True
                # if by end of set comparison current individual was never dominated, it belongs in the rank set
                if not dominated and population[j, self._rank_index] == not_ranked:
                    population[j, self._rank_index] = rank

            # check if completed
            complete = True
            for ind in range(population_size):
                if population[ind, self._rank_index] == not_ranked:
                    complete = False

        # congratulations, you made it this far, sort the solutions and move on with your life
        self._max_rank = int(max(population[:, self._rank_index]))
        population = population[population[:, self._rank_index].argsort()]
        return population

    def crowded_rank(self, population: np.ndarray) -> np.ndarray:
        """
        Calculate the crowded ranking for a population based on each pareto rank. This sorts
        solutions again from within a pareto rank.
        :param population: a population of individuals
        :return: ranked population array
        """
        population_size = np.shape(population)[0]
        population = population[population[:, self._f1_index].argsort()]
        # break the population into ranked sets
        for rank in range(1, self._max_rank + 1):
            # how many individuals exist in the current rank?
            num_members = np.count_nonzero(population[:, self._rank_index] == rank)
            # create index list of all individuals in the current pareto rank
            indices = [ind for ind in range(population_size) if int(population[ind, self._rank_index]) == rank]

            # set the first and last individuals to have a very large crowding distance
            max_f1 = self._fitness_bounds['max_f1']
            min_f1 = self._fitness_bounds['min_f1']
            max_f2 = self._fitness_bounds['max_f2']
            min_f2 = self._fitness_bounds['min_f2']
            max_crowding_score = max_f1 - min_f1 + max_f2 - min_f2
            population[indices[0], self._crowding_index] = max_crowding_score
            population[indices[-1], self._crowding_index] = max_crowding_score

            # Set middle individuals crowding distance to be a function of neighbours cuboid separation
            for ind in range(1, num_members - 1):
                a_fitness_1 = population[indices[ind - 1], self._f1_index]
                a_fitness_2 = population[indices[ind - 1], self._f2_index]
                b_fitness_1 = population[indices[ind + 1], self._f1_index]
                b_fitness_2 = population[indices[ind + 1], self._f2_index]
                crowding_score = abs(a_fitness_1 - b_fitness_1) + abs(a_fitness_2 - b_fitness_2)
                population[indices[ind], self._crowding_index] = crowding_score
        return population

    @staticmethod
    def derangement(length: int) -> list:
        """
        WARNING: here be dragons. You would need to be a deranged lunatic to modify this in any way.

        This function creates a shuffled list where each element is guaranteed
        to not be what it was initially (not truly random shuffle). Only Conrad knew how this worked once long
        ago, but now even that knowledge may be lost forever. Enter at your own risk.

        :param length: the length of the derangement to create
        :return: a new shuffled list
        """
        while True:
            v = np.linspace(0, length - 1, length, dtype=int)
            for j in range(length - 1, -1, -1):
                p = rd.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return list(v)

    def competition(self, individual_1: list, individual_2: list) -> list:
        """
        compare two individuals and return the more fit one. First sorted by pareto rank, and then
        by largest crowding within a specific rank
        :param individual_1: first individual
        :param individual_2: second individual
        :return: individual
        """
        if individual_1[self._rank_index] < individual_2[self._rank_index]:
            return individual_1
        elif individual_1[self._rank_index] > individual_2[self._rank_index]:
            return individual_2
        else:
            if individual_1[self._crowding_index] > individual_2[self._crowding_index]:
                return individual_1
            else:
                return individual_2

    def plot_pareto(self) -> None:
        """
        plot the pareto front
        :return: None
        """
        plt.clf()
        index = 0
        for rank in range(1, int(self._max_rank + 1)):
            # how many individuals in each rank
            num_members = np.count_nonzero(self._population[:, self._rank_index] == rank)
            # pull out fitness data along each axis of current front
            f1 = self._population[index:index + num_members, self._f1_index]
            f2 = self._population[index:index + num_members, self._f2_index]
            # combine and sort fitness data to plot a line
            f = np.vstack((f1, f2))
            f = f[:, np.argsort(f[1])]
            # update index for next rank
            index = index + num_members
            # create4 graph and plot the line
            plt.plot(f[0, :], f[1, :], linestyle='--', marker='o', label='Front: ' + str(rank))
        plt.xlabel(self._f1_plot_label)
        plt.ylabel(self._f2_plot_label)
        plt.title('Fitness progress, Generation: {}'.format(self._current_generation))
        plt.legend(loc=1)
        save_path = os.path.join(self._plots_directory, '{}_generation{}.png'.format(self._filename,
                                                                                     self._current_generation))
        plt.savefig(save_path, bbox_inches='tight')
        plt.show(block=False)
        plt.draw()
        plt.pause(.005)

    def save_data(self, filepath: str) -> None:
        """
        save the current generations data
        :param filepath:
        :return: None
        """
        np.save(filepath, self._population_history)

    def run(self) -> None:
        """
        Main algorithm running method
        :return: None
        """
        # setup the initial generation
        self._population = self.create_population()
        self._population = self.evaluate_population(self._population)
        self._population = self.pareto_rank(self._population)
        self._population = self.crowded_rank(self._population)
        self.plot_pareto()
        self._population_history[0, :, :] = self._population

        # run for all generations
        for generation in range(2, self._total_generations + 1):
            # update the current generation
            self._current_generation = generation

            # create a mating pool copy of the population
            mating_pool = np.zeros(shape=np.shape(self._population))
            for individual in range(self._population_size):
                for parameter in range(self._individual_size):
                    mating_pool[individual, parameter] = self._population[individual, parameter]

            # Using crowded tournament selection on P, create mating pool of elites of size N
            mating_pool = self.crowded_rank(mating_pool)

            # Using mating population and blended crossover create child population C of size N
            child_population = self.blended_n_point_crossover(mating_pool)

            # Apply mutation to child population C
            child_population = self.mutation(child_population)

            # Evaluate performance in both objectives for each member of population C
            child_population = self.evaluate_population(child_population)

            # Add child population to parent population, yielding population R of size 2N
            full_population = np.concatenate((self._population, child_population))

            # Evaluate pareto front rank F of each member in R
            full_population = self.pareto_rank(full_population)

            # Evaluate crowding value Ci of each member of R within their rank F
            full_population = self.crowded_rank(full_population)

            # Sort population R of size 2N first by F, then by Ci. Remove lower half of population
            full_population = self.slice(full_population)
            for individual in range(self._population_size):
                for parameter in range(self._individual_size):
                    self._population[individual, parameter] = full_population[individual, parameter]

            # save it
            self._population_history[generation - 1, :, :] = self._population
            if self._save_data:
                self.save_data(self.filename)

            # plot it
            self.plot_pareto()


if __name__ == '__main__':
    rd.seed(1)
    parameters = [(-5, 5, False), (-5, 5, False), (-5, 5, False)]
    size = 100
    num_generations = 50
    ga = MultiObjectiveGeneticAlgorithm(parameters, size, num_generations, get_kursawe_fitness)
    ga.mutation_multiplier = 0.2
    ga.run()
