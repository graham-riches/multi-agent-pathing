"""
    @file test_main.py
    @brief run all the project unit tests
    @author Graham Riches
    @details
   
"""

import unittest


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(test_suite)
