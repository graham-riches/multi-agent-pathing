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
import time
import os
import random as rd
import matplotlib.pylab as plt
import sys
from optimization.kursawe import Kursawe


class Parameter:
    def __init__(self, lower_limit: float, upper_limit: float, is_integer: bool) -> None:
        """
        Parameter type for multi-objective genetic algorithm
        :param lower_limit: the parameters lower bound
        :param upper_limit: parameters upped bound
        :param is_integer: cast value to integer?
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.is_int = is_integer


class MultiObjectiveGeneticAlgorithm:
    def __init__(self, parameters: list, population_size: int,
                 generations_to_run: int, fitness_function: callable) -> None:
        """
        Create an instance of the MOEA algorithm. NOTE: most of this algorithm is done using 2-dimensional arrays
        and has been optimized for speed, hence a few things may be somewhat hard to understand! The biggest one
        is that for each candidate in a population, the fitness, ranking, and crowding distances are tacked
        onto the end of the parameters list in the array. This makes it trivial-ish and relatively fast to sort the
        population by its fitness ranking and whatnot. So beware of that ¯\_(ツ)_/¯

        :param parameters: list of Parameter objects. These have bounded ranges and type specification. NOTE:
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
        self._parameter_upper_bounds = [parameter.upper_limit for parameter in parameters]
        self._parameter_lower_bounds = [parameter.lower_limit for parameter in parameters]
        self._parameter_is_integer = [parameter.is_int for parameter in parameters]
        self._f1_index = self._total_parameters
        self._f2_index = self._total_parameters + 1
        self._rank_index = self._total_parameters + 2
        self._crowding_index = self._total_parameters + 3

        # simulation, generation, population etc.
        self._population_size = int(population_size) if population_size % 2 == 0 else int(population_size + 1)
        self._total_generations = int(generations_to_run) if generations_to_run >= 1 else 1
        self._current_generation = 0
        self._population = None  # 2D array of individuals associated with the current generation
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
        self._filename = 'test'
        self._plots_directory = 'PLOTS'  # create if it does not exist
        if not os.path.exists(self._plots_directory):
            os.mkdir(self._plots_directory)

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

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        self._filename = filename

    def create_population(self) -> None:
        """
        Create the GA population
        :return: None
        """
        self._population = np.zeros(shape=(self._population_size, self._individual_size))
        self._population_history = np.zeros(shape=(self._total_generations,
                                                   self._population_size,
                                                   self._individual_size))
        # create a random population within the specified parameter bounds
        for individual in range(self._population_size):
            for parameter in range(self._total_parameters):
                parameter_range = self._parameter_upper_bounds[parameter] - self._parameter_lower_bounds[parameter]
                self._population[individual, parameter] = rd.random() * parameter_range + self._parameter_lower_bounds[parameter]
        # round if required
        for parameter in range(self._total_parameters):
            if self._parameter_is_integer[parameter]:
                self._population[:, parameter] = self._population[:, parameter].round()

    def evaluate_population(self) -> None:
        """
        Evaluate the current population by calculating the fitness of each individual and ranking them
        :return: None
        """
        # run through all individuals and calculate their fitness
        for individual in range(self._population_size):
            for parameter in range(self._total_parameters):
                # cast integer parameters if required
                if self._parameter_is_integer[parameter]:
                    self._population[individual, parameter] = round(self._population[individual, parameter], 0)
            # calculate fitness
            f1, f2 = self._fitness_function(self._population[individual, 0:self._total_parameters])
            # assign the fitness values to the 2D population array for the current individual
            self._population[individual, self._f1_index] = f1
            self._population[individual, self._f2_index] = f2
        # get the bounds for the current population
        self._fitness_bounds['min_f1'] = min(self._population[:, self._f1_index])
        self._fitness_bounds['max_f1'] = max(self._population[:, self._f1_index])
        self._fitness_bounds['min_f2'] = min(self._population[:, self._f2_index])
        self._fitness_bounds['max_f2'] = max(self._population[:, self._f2_index])

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

    def crowded_tournament(self):
        pass

    def blended_n_point_crossover(self):
        pass

    def mutation(self):
        pass

    def slice(self):
        pass

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
            individual = 0
            found_unranked = False
            while not found_unranked:
                if population[individual, self._rank_index] == not_ranked:
                    population[individual, self._rank_index] = rank
                    found_unranked = True
                else:
                    individual += 1

            # check all other unranked solutions against entire unranked or current ranked set
            for check_individual in range(individual, population_size):
                # check to see if this individual ever gets dominated by another
                dominated = False
                for other_individual in range(individual, check_individual):
                    # ensure genome has not yet been ranked and its comparator is unranked or in the current rank set
                    if (population[check_individual, self._rank_index] == not_ranked and
                        (population[other_individual, self._rank_index] == not_ranked or
                         population[other_individual, self._rank_index] == rank)):
                        # if current individual dominates comparator, set comparators rank to -1
                        if self.dominates(population[check_individual, :], population[other_individual]):
                            population[other_individual, self._rank_index] = not_ranked
                        else:
                            dominated = True
                    # if by end of set comparison current individual was never dominated, it belongs in the rank set
                    if not dominated and population[check_individual, self._rank_index] == not_ranked:
                        population[check_individual, self._rank_index] = rank

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

    def crossover_index(self):
        pass

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
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Fitness progress, Generation: {}'.format(self._current_generation))
        plt.legend(loc=1)
        save_path = os.path.join(self._plots_directory, 'Generation{}.png'.format(self._current_generation))
        plt.savefig(save_path, bbox_inches='tight')
        plt.show(block=False)
        plt.draw()
        plt.pause(.005)

    def initialize(self) -> None:
        """
        initialize the algorithm by creating a population and doing the initial sortation on it
            1. create population
            2. evaluate fitness
            3. pareto rank
            4. calculate crowding distance

        :return: None
        """
        self.create_population()
        self.evaluate_population()
        self._population = self.pareto_rank(self._population)
        self._population = self.crowded_rank(self._population)

    def run(self) -> None:
        """
        Main algorithm running method
        :return: None
        """
        self.initialize()
        self.plot_pareto()

















def MOEA(PopulationSize, GenerationsToRun, MutationMultiplier):
    class Parameters:
        # Set variable limits (Hard Constraint)
        up = [5, 5, 5]  # parameter upper limits
        lo = [-5, -5, -5]  # parameter lower limits
        iGen = [0, 0, 0]  # Define which variables are to be cast as integers
        ActiveParameters = 0  # number of parameters we are actually evolving (Set to 0 for automatic)
        MutUniformity = 1  # Mutation non-uniformity, ~.5 is low, ~4 is high
        bCross = .5  # percent chance that a crossover will be blended between parents
        nCross = 3  # maximum number of crossovers applied per child
        de = True  # Show debug tips
        xde = False  # Show exhaustive debug mesages (slows down process quite a bit)
        testName = 'test'  # filename header for saved data
        # DO NOT MODIFY FROM HERE ON
        gSize = len(up) + 4  # Genome plus f1, f2, Ri, Distance
        Cgen = 1  # current generations
        pop = PopulationSize
        Tgen = GenerationsToRun
        Mut = MutationMultiplier  # mutation multiplier, default is 1/n, where n is number of genes
        maxRank = 0
        minf1 = 0
        minf2 = 0
        maxf1 = 0
        maxf2 = 0
        fileLoc = os.getcwd()  # folder location to store data

    # create the stored images location if it doesn't exist yet
    if not os.path.exists('PLOTS'):
        os.mkdir('PLOTS')

    # Debug options
    global History
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    CheckInput(Parameters, PopulationSize, GenerationsToRun, MutationMultiplier)

    # Create random population P of size N
    History, Genome = CreatePopulation(Parameters)

    # Evaluate performance in both objectives for each member in P
    Genome = EvaluatePopulation(Parameters, Genome)

    # Evaluate pareto front rank F of each member in P
    Genome = ParetoRank(Parameters, Genome)

    # Evaluate crowding value Ci of each member in P within their rank F
    Genome = CrowdedRank(Parameters, Genome)

    # Visualize data
    Visualization(Parameters, Genome)

    # For g=1:generations
    for g in range(2, GenerationsToRun + 1):
        Parameters.Cgen = g
        # Using crowded tournament selection on P, create mating pool of elites of size N
        MatingPool = CrowdedTournament(Parameters, Genome)

        # Using mating population and blended crossover create child population C of size N
        GenomeChild = BlendedNPointCrossover(Parameters, MatingPool)

        # Apply mutation to child population C
        GenomeChild = Mutation(Parameters, GenomeChild)

        # Evaluate performance in both objectives for each member of population C
        GenomeChild = EvaluatePopulation(Parameters, GenomeChild)

        # Add child population to parent population, yielding population R of size 2N
        GenomeFull = np.concatenate((Genome, GenomeChild))

        # Evaluate pareto front rank F of each member in R
        t = time.time()
        GenomeFull = ParetoRank(Parameters, GenomeFull)
        elapsed = time.time() - t
        print(elapsed)

        # Evaluate crowding value Ci of each member of R within their rank F
        GenomeFull = CrowdedRank(Parameters, GenomeFull)

        # Sort population R of size 2N first by F, then by Ci. Remove lower half of population, creating population P of size N
        Genome = Slice(Parameters, GenomeFull)

        # Save genome data in master history matrix
        History[g - 1, :, :] = SaveData(Parameters, Genome, History)

        # Visualize data
        Visualization(Parameters, Genome)


def CrowdedTournament(Parameters, Genome):
    # We want every solution to participate in the tournament twice, paired with
    # random other entity, without ever pairing with itself, with a perfectly
    # uniform probability. This requires that a uniform derangement be performed
    # on the indice list
    TestList = Derangement(Parameters.pop)
    # Create empty mating pool array
    MatingPool = Genome * 0
    # compare each individual with its opponent
    for i in range(Parameters.pop):
        MatingPool[i, :] = Competition(Parameters, Genome[i, :], Genome[TestList[i], :])
    if Parameters.xde:
        print("MATING POOL:")
        print(MatingPool)
        sys.stdout.flush()
    return MatingPool


def BlendedNPointCrossover(Parameters, MatingPool):
    # Create an empty matrix to store completed child genomes in
    GenomeChild = MatingPool * 0
    # Create each individual by selecting two random parents from the mating pool
    for i in range(Parameters.pop):
        # Choose two random, unique parents
        Rand1 = rd.randint(0, Parameters.pop - 1)
        check = True
        while check == True:
            Rand2 = rd.randint(0, Parameters.pop - 1)
            if Rand1 != Rand2:
                check = False
        # Select parents from mating pool
        Parents = np.zeros([2, Parameters.gSize])
        Parents[0, :] = MatingPool[Rand1, :]
        Parents[1, :] = MatingPool[Rand2, :]
        # Create an index of the crossover locations
        CrossPoints = CrossoverIndex(Parameters)
        # go through each gene and choose from correct parent
        Parent = 1
        for j in range(Parameters.gSize - 4):
            # change parent at crossover points
            if CrossPoints[j] == 1:
                if Parent == 1:
                    Parent = 2
                else:
                    Parent = 1
            # place parent value into offspring, potentially blend parents on
            # crossover points
            # BLENDING CASE
            if Parameters.bCross > rd.random() and CrossPoints[j] > 0:
                # choose random blend amount
                Blend = rd.random()
                GenomeChild[i, j] = Blend * Parents[0, j] + (1 - Blend) * Parents[1, j]
            # NON BLENDING CASE
            else:
                GenomeChild[i, j] = Parents[Parent - 1, j]
    if Parameters.xde:
        print("CHILD POPULATION:")
        print(GenomeChild)
        sys.stdout.flush()
    return GenomeChild


def Mutation(Parameters, Genome):
    # chance to mutate any given gene
    if Parameters.ActiveParameters == 0:
        scale = Parameters.gSize - 4
    else:
        scale = Parameters.ActiveParameters
    MutChance = (1.0 / scale) * Parameters.Mut
    # go through each member of the population
    for i in range(Parameters.pop):
        # go through each gene in each individual
        for j in range(Parameters.gSize - 4):
            # does mutation occur?
            if rd.random() < MutChance:
                tau = rd.randint(0, 1)
                GenRatio = Parameters.Cgen / Parameters.Tgen
                Multiplier = (1 - rd.random() ** ((1 - GenRatio) ** Parameters.MutUniformity))
                # use non-uniform mutator (Michalewicz, 1992)
                if tau == 1:
                    Genome[i, j] = Genome[i, j] + (Parameters.up[j] - Genome[i, j]) * Multiplier
                else:
                    Genome[i, j] = Genome[i, j] - (Genome[i, j] - Parameters.lo[j]) * Multiplier
    if Parameters.xde:
        print("MUTATED POPULATION:")
        print(Genome)
        sys.stdout.flush()
    return Genome


def Slice(Parameters, Genome):
    # order array first by rank, then by crowding distance
    idx = np.lexsort((Genome[:, Parameters.gSize - 1] * -1, Genome[:, Parameters.gSize - 2]))
    Genome = Genome[idx]
    Parameters.maxRank = max(Genome[0:Parameters.pop, Parameters.gSize - 2])
    if Parameters.xde:
        print("RETAINED POPULATION:")
        print(Genome[0:Parameters.pop, :])
        sys.stdout.flush()
    return Genome[0:Parameters.pop, :]


def CrossoverIndex(Parameters):
    # create array of ones and zeros, where a one is the index of a crossover
    # array has to be random, and contain n crossover points
    CrossPoints = np.random.rand(Parameters.gSize - 4)
    # extract maximum n values, where n is number of crossovers, (1-nCross)
    n = rd.randint(1, Parameters.nCross)
    ind = CrossPoints.argsort()[-n:][::-1]
    CrossPoints = CrossPoints * 0
    CrossPoints[ind] = 1
    return CrossPoints


def Visualization(Parameters, Genome):
    # It would be nice to see the entire population graphed on an f1 vs f2 plane
    # with every pareto front drawn as a line. First step is to separate pareto
    # fronts
    plt.clf()
    index = 0
    for rank in range(1, int(Parameters.maxRank + 1)):
        # how many individuals in each rank
        NumMembers = np.count_nonzero(Genome[:, Parameters.gSize - 2] == rank)
        # pull out fitness data along each axis of current front
        f1 = Genome[index:index + NumMembers, Parameters.gSize - 4]
        f2 = Genome[index:index + NumMembers, Parameters.gSize - 3]
        # combine and sort fitness data to plot a line
        f = np.vstack((f1, f2))
        f = f[:, np.argsort(f[1])]
        # update index for next rank
        index = index + NumMembers
        # create4 graph and plot the line
        plt.plot(f[0, :], f[1, :], linestyle='--', marker='o', label='Front: ' + str(rank))
    plt.xlabel('COST OF STRUCTURE')
    plt.ylabel('SECONDS PER PRESENTATION')
    plt.title('Fitness progress, Generation: ' + str(Parameters.Cgen))
    plt.legend(loc=1)
    plt.savefig('PLOTS\Generation' + str(Parameters.Cgen) + '.png', bbox_inches='tight')
    plt.show(block=False)
    plt.draw()
    plt.pause(.005)


def SaveData(Parameters, Genome, History):
    filename0 = Parameters.fileLoc + '\\Last_History.npy'
    filename = Parameters.fileLoc + '\\' + Parameters.testName + '_History.csv'
    filename2 = Parameters.fileLoc + '\\' + Parameters.testName + '_History.npy'
    np.save(filename0, History)
    np.save(filename2, History)
    with open(filename, 'a') as f_handle:
        np.savetxt(f_handle, Genome, delimiter=",")
    return Genome


def Review(History):
    # It would be nice to see the entire population graphed on an f1 vs f2 plane
    # with every pareto front drawn as a line. First step is to separate pareto
    # fronts
    a = np.array([[1, 2, 3], [1, 2, 3]])

    # Use History=0 for automatic load from C:\Users\Conrad\Documents\PYTHON\History.npy
    if type(History) != type(a):
        History = np.load(os.getcwd() + '\\Last_History.npy')
    DisplayTime = input("Seconds between generations?")
    Shape = np.shape(History)
    Generations = Shape[0]
    gSize = Shape[2]
    # Find min and max fitnesses for graph range
    maxf1 = np.zeros(Generations)
    minf1 = np.zeros(Generations)
    maxf2 = np.zeros(Generations)
    minf2 = np.zeros(Generations)
    for i in range(Generations):
        Genome = History[i, :, :]
        maxf1[i] = max(Genome[:, gSize - 4])
        maxf2[i] = max(Genome[:, gSize - 3])
        minf1[i] = min(Genome[:, gSize - 4])
        minf2[i] = min(Genome[:, gSize - 3])
    Maxf1 = max(maxf1[np.nonzero(maxf1)])
    Minf1 = min(minf1[np.nonzero(minf1)])
    Maxf2 = max(maxf2[np.nonzero(maxf2)])
    Minf2 = min(minf2[np.nonzero(minf2)])
    xRange = Maxf1 - Minf1
    yRange = Maxf2 - Minf2
    for i in range(Generations - 1):
        Genome = History[i, :, :]
        maxRank = max(Genome[:, gSize - 2])
        plt.clf()
        index = 0
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Fitness progress, Generation: ' + str(i))
        plt.xlim([Minf1 - xRange * .1, Maxf1 + xRange * .1])
        plt.ylim([Minf2 - yRange * .1, Maxf2 + yRange * .1])
        for rank in range(1, int(maxRank + 1)):
            # how many individuals in each rank
            NumMembers = np.count_nonzero(Genome[:, gSize - 2] == rank)
            # pull out fitness data along each axis of current front
            f1 = Genome[index:index + NumMembers, gSize - 4]
            f2 = Genome[index:index + NumMembers, gSize - 3]
            # combine and sort fitness data to plot a line
            f = np.vstack((f1, f2))
            f = f[:, np.argsort(f[1])]
            # update index for next rank
            index = index + NumMembers
            # create4 graph and plot the line
            plt.plot(f[0, :], f[1, :], linestyle='--', marker='o', label='Front: ' + str(rank))
        # plt.pause(.0001)
        plt.legend(loc=1)
        plt.show()
        # plt.draw()
        time.sleep(float(DisplayTime))


def CreateAndRunNexusSim(x):
    # f1 is cost of solution
    # f2 is presentation efficiency
    f1 = x[0] * 50000 + x[1] * x[2] * x[3] * 1000 + x[5] * 8000
    c1 = max([x[0] - (x[0] * .2) ** 2, 0])
    c2 = max([x[1] - (x[1] * .2) ** 2, 0])
    c3 = max([x[2] - (x[2] * .3) ** 2, 0])
    c4 = max([x[3] - (x[3] * .1) ** 4, 0])
    c5 = max([x[4] - (x[4] * .2) ** 4, 0])
    c6 = x[5] * 10
    f2 = 500 / (c1 + c2 + c3 + c4 + c5 + c6)
    return f1, f2


def get_kursawe_fitness(x_values: list) -> tuple:
    kursawe = Kursawe(x_values)
    return kursawe.f1, kursawe.f2


if __name__ == '__main__':
    MOEA(100, 50, 0.2)
