from tomomak.test_objects import objects2d
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import *
import inspect
import unittest


class TestModel(unittest.TestCase):

    def test__all_functions_defaults(self):
        axes = [Axis1d(), Axis1d()]
        mesh = Mesh(axes)
        for tested_method in [o for o in inspect.getmembers(objects2d) if inspect.isfunction(o[1])]:
            tested_method[1](mesh)