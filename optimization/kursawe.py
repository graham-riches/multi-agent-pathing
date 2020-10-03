"""
    @file kursawe.py
    @brief kursawe function for 2d optimization for testing out the GA implementation
    @author Graham Riches
    @details
        This uses the multi-objective benchmark function known as the Kursawe function to
        test the GA implementation to make sure it works. Details here:
            https://en.wikipedia.org/wiki/Test_functions_for_optimization
        The objective is to minimize the output of two fitness variables that are given by fairly
        heinous functions.

        This class is designed to just be a single instance calculation for a given set of parameters.
"""
import numpy as np


class Kursawe:
    def __init__(self, x_values: list) -> None:
        """
        Creates a single point instance of the Kurwsawe function for calculating multi-objective
        fitness. x_values is a list of length three that contains three weights that are used
        to calculate the fitness values f1 and f2
        :param x_values: list of input parameters
        """
        self._x = x_values

    @property
    def f1(self) -> float:
        return self.calculate_f1()

    @property
    def f2(self) -> float:
        return self.calculate_f2()

    def calculate_f1(self) -> float:
        """
        calculate the first fitness value based on the input parameters
        NOTE: the functions are absolutely disgusting, so this may be gross code ¯\_(ツ)_/¯
        :return: floating point output value
        """
        fitness_value = 0
        for i in range(2):
            x_i = self._x[i]
            x_i1 = self._x[i + 1]
            exp_arg = -0.2 * np.sqrt(x_i**2 + x_i1**2)
            fitness_value += -10 * np.exp(exp_arg)
        return fitness_value

    def calculate_f2(self) -> float:
        """
        calculate the second fitness value based on the input parameters
        NOTE: the functions are absolutely disgusting, so this may be gross code ¯\_(ツ)_/¯
        :return: floating point output value
        """
        fitness_value = 0
        for i in range(3):
            x_i = self._x[i]
            fitness_value += abs(x_i)**0.8 + 5 * np.sin(x_i**3)
        return fitness_value


def get_kursawe_fitness(x_values: list) -> tuple:
    """
    calculate the multi-objective fitness for a kursawe object
    :param x_values:
    :return:
    """
    kursawe = Kursawe(x_values)
    return kursawe.f1, kursawe.f2