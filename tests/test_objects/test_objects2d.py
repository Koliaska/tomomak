from tomomak.test_objects import objects2d
from tomomak.mesh.mesh import *
from tomomak.mesh import cartesian
import inspect
import unittest


class TestModel(unittest.TestCase):

    def test__all_functions_defaults(self):
        axes = [cartesian.Axis1d(upper_limit=8, size=3),  cartesian.Axis1d(upper_limit=10, size=5)]
        mesh = Mesh(axes)
        for tested_method in [o for o in inspect.getmembers(objects2d) if inspect.isfunction(o[1])]:
            tested_method[1](mesh)