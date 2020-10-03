"""
    @file test_kursawe.py
    @brief tests to make sure the Kursawe class does sketchy math properly
    @author Graham Riches
    @details
   
"""
import unittest
from optimization.kursawe import Kursawe


class TestKursawe(unittest.TestCase):
    def setUp(self):
        x_values = [-1, 2, 2]
        self.kursawe = Kursawe(x_values)

    def tearDown(self):
        pass

    def test_f1(self):
        """ shady floating point comparison to hand calculation to verify """
        fitness = self.kursawe.f1
        self.assertAlmostEqual(-12.07378, fitness, places=4)

    def test_f2(self):
        """ shady floating point comparison to hand calculation to verify """
        fitness = self.kursawe.f2
        self.assertAlmostEqual(10.1684, fitness, places=4)
