from tomomak.mesh.cartesian import *
import unittest
from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
import tomomak.constraints.basic
import scipy.ndimage

class TestBasic(unittest.TestCase):

    def test__apply_along_axis(self):
        """Test that apply along axis works correctly with sorted.
        """
        axes = [Axis1d(name="x", units="cm", size=3, upper_limit=10)]
        mesh = Mesh(axes)
        solution = np.array([0, 4, 2])
        det = np.array([[0, 0, 0]])
        det_signal = np.array([0])
        mod = Model(mesh=mesh, detector_signal=det_signal, detector_geometry=det, solution=solution)
        solver = Solver()
        steps = 1
        c = tomomak.constraints.basic.ApplyAlongAxis(sorted, axis=0, alpha=1)
        solver.constraints = [c]
        solver.solve(mod, steps=steps)
        assert all(np.diff(mod.solution) >= 0)

    def test__apply_function(self):
        """Test that ApplyFunction works with Gaussian filter.
        """
        axes = [Axis1d(name="x", units="cm", size=2,upper_limit=10),
                Axis1d(name="x", units="cm", size=3, upper_limit=10)]
        mesh = Mesh(axes)
        solution = np.array([[1, 1, 1], [1, 1, 1]])
        det = np.array([[[0, 0, 0], [0, 0, 0]]])
        det_signal = np.array([0])
        mod = Model(mesh=mesh, detector_signal=det_signal, detector_geometry=det, solution=solution)
        solver = Solver()
        steps = 1
        c = tomomak.constraints.basic.ApplyFunction(scipy.ndimage.gaussian_filter, sigma=1, alpha=1)
        solver.constraints = [c]
        solver.solve(mod, steps=steps)
        assert not np.all(np.diff(mod.solution))



