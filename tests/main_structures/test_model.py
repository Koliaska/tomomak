from tomomak import model
import numpy as np
import unittest


class TestModel(unittest.TestCase):

    def test__different_size_geometry_signal(self):
        with self.assertRaises(Exception):
            geometry = np.zeros((10, 5))
            signal = np.zeros(5)
            model.Model(geometry, signal)

    def test__check_equal_size_geometry_signal(self):
        geometry = np.zeros((10, 5))
        signal = np.zeros(10)
        model.Model(geometry, signal)

    def test__different_size_geometry_solution(self):
        with self.assertRaises(Exception):
            geometry = np.zeros((10, 5, 3))
            solution = np.zeros((5, 4))
            model.Model(geometry, None, solution)

    def test__equal_size_geometry_solution(self):
        geometry = np.zeros((10, 5, 3))
        solution = np.zeros((5, 3))
        model.Model(geometry, None, solution)

    def test__signal_is_2D(self):
        with self.assertRaises(Exception):
            geometry = np.zeros((10, 5))
            signal = np.zeros((10, 5))
            model.Model(geometry, signal)

    def test__signal_is_not_a_number(self):
        with self.assertRaises(Exception):
            geometry = np.zeros((2, 5))
            signal = ('a', 'a')
            model.Model(geometry, signal)
