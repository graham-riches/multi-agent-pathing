"""
    @file test_genetic_algorithm.py
    @brief unit tests for the multi-object evolutionary algorithm
    @author Graham Riches
    @details
        Some tests for making sure the GA works as intended :)
"""
import unittest
import numpy as np
from optimization.genetic_algorithm import MultiObjectiveGeneticAlgorithm, Parameter
from optimization.kursawe import get_kursawe_fitness


class TestMultiObjectiveGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self._population_size = 20
        self._generations = 10
        parameters = [Parameter(-5, 5, False), Parameter(-5, 5, False), Parameter(-5, 5, False)]
        self._parameters = len(parameters)
        self.ga = MultiObjectiveGeneticAlgorithm(parameters, self._population_size,
                                                 self._generations, get_kursawe_fitness)
        self.ga._mutation_multiplier = 0.6
        self.population = self.ga.create_population()

    def tearDown(self):
        self.ga = None

    def test_derangement(self):
        derangement = self.ga.derangement(10)
        for i in range(10):
            self.assertTrue(derangement[i] != i)

    def test_create_population(self):
        self.assertTupleEqual((self._population_size, self._parameters + 4), np.shape(self.population))
        self.assertTupleEqual((self._generations, self._population_size, self._parameters + 4),
                              np.shape(self.ga._population_history))
        for i in range(self._population_size):
            individual = self.population[i, :]
            for j in range(self.ga._total_parameters):
                above_lower = individual[j] >= self.ga._parameter_lower_bounds[j]
                below_upper = individual[j] <= self.ga._parameter_upper_bounds[j]
                self.assertTrue(above_lower and below_upper)

    def test_evaluate_generation(self):
        self.ga.evaluate_population(self.population)
        for individual in range(self._population_size):
            self.assertGreater(abs(self.population[individual, self._parameters]), 0)
            self.assertGreater(abs(self.population[individual, self._parameters + 1]), 0)

        for key, value in self.ga._fitness_bounds.items():
            self.assertGreater(abs(value), 0)

    def test_dominates(self):
        # make some fake individuals to test the dominates function
        f1_index = self._parameters
        f2_index = self._parameters + 1
        i1 = np.zeros(self._parameters + 4)
        i2 = np.zeros(self._parameters + 4)
        # test scenarios
        fitness_scenarios = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
        for scenario in fitness_scenarios:
            i1[f1_index] = scenario[0]
            i1[f2_index] = scenario[1]
            i2[f1_index] = scenario[2]
            i2[f2_index] = scenario[3]
            self.assertTrue(self.ga.dominates(i1, i2))
            self.assertFalse(self.ga.dominates(i2, i1))

    def test_pareto_rank(self):
        population = self.ga.evaluate_population(self.population)
        population = self.ga.pareto_rank(population)
        rank_index = self._parameters + 2
        current_rank_index = 1
        for individual in range(self._population_size):
            rank = int(population[individual, rank_index])
            self.assertGreaterEqual(rank, current_rank_index)
            if rank >= current_rank_index:
                current_rank_index = rank

    def test_calculate_crowding_score(self):
        population = self.ga.create_population()
        self.ga.evaluate_population(population)
        rank_index = self._parameters + 2
        crowding_index = self._parameters + 3
        population = self.ga.pareto_rank(population)
        population = self.ga.crowded_rank(population)
        # for each unique rank, make sure that all middle members have values less than the outliers
        unique_ranks = np.unique(population[:, rank_index])
        for rank in unique_ranks:
            indices = [ind for ind in range(self._population_size) if population[ind, rank_index] == rank]
            self.assertAlmostEqual(population[indices[0], crowding_index],
                                   population[indices[-1], crowding_index], places=2)
            max_crowding = population[indices[0], crowding_index]
            if len(indices) > 2:
                for i in range(1, len(indices) - 1):
                    self.assertLess(population[indices[i], crowding_index], max_crowding)

    def test_competition(self):
        population = self.ga.create_population()
        self.ga.evaluate_population(population)
        rank_index = self._parameters + 2
        crowding_index = self._parameters + 3
        # make some fake individuals to test the competition
        i1 = np.zeros(self._parameters + 4)
        i2 = np.zeros(self._parameters + 4)
        # test scenarios
        fitness_scenarios = [[0, 5, 0, 4], [0, 0, 2, 0]]
        for scenario in fitness_scenarios:
            i1[rank_index] = scenario[0]
            i1[crowding_index] = scenario[1]
            i2[rank_index] = scenario[2]
            i2[crowding_index] = scenario[3]
            selected = self.ga.competition(i1, i2)
            for ind, param in enumerate(selected):
                self.assertEqual(i1[ind], param)

    def test_crowded_tournament(self):
        # this one is tough to evaluate, so just ensure the size is what we expect
        self.ga.evaluate_population(self.population)
        population = self.ga.pareto_rank(self.population)
        population = self.ga.crowded_rank(population)
        pop_size = np.shape(population)
        population = self.ga.crowded_tournament(population)
        self.assertTupleEqual(pop_size, np.shape(population))

    def test_crossover_points(self):
        points = self.ga.crossover_index()
        # count total points should be less than or equal to max child crossovers
        total = sum(points)
        self.assertLessEqual(total, self.ga.max_crossovers)

    def test_slice(self):
        population = self.ga.evaluate_population(self.population)
        population = self.ga.pareto_rank(population)
        population = self.ga.crowded_rank(population)
        pop_size = np.shape(population)
        population = self.ga.crowded_tournament(population)
        big_pop = np.concatenate((population, population))
        new_pop = self.ga.slice(big_pop)
        self.assertTupleEqual(pop_size, np.shape(new_pop))

    def test_mutation(self):
        compare_pop = np.zeros(np.shape(self.population))
        for individual in range(self._population_size):
            for gene in range(self._parameters):
                compare_pop[individual, gene] = self.population[individual, gene]
        new_population = self.ga.mutation(self.population)
        all_the_same = True
        for individual in range(self._population_size):
            for gene in range(self._parameters):
                if new_population[individual, gene] != compare_pop[individual, gene]:
                    all_the_same = False
        self.assertFalse(all_the_same)

    def test_blended_crossover(self):
        self.ga._current_generation = 1
        population = self.ga.create_population()
        compare_pop = np.zeros(np.shape(population))
        for individual in range(self._population_size):
            for gene in range(self._parameters):
                compare_pop[individual, gene] = population[individual, gene]
        new_population = self.ga.blended_n_point_crossover(population)
        all_the_same = True
        for individual in range(self._population_size):
            for gene in range(self._parameters):
                if new_population[individual, gene] != compare_pop[individual, gene]:
                    all_the_same = False
        self.assertFalse(all_the_same)
