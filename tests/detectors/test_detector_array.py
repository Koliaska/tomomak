from tomomak import model
import unittest
from tomomak.mesh import mesh
from tomomak.mesh import cartesian


class TestDetectorArray(unittest.TestCase):
    def test__sanity(self):
        axes = [cartesian.Axis1d(name="X", units="cm", size=2, upper_limit=10),
                cartesian.Axis1d(name="Y", units="cm", size=2, upper_limit=10)]
        m = mesh.Mesh(axes)
        mod = model.Model(mesh=m)
        from tomomak.detectors import detector_array
        kw_list = [dict(mesh=m, p1=(-5, 0), p2=(15, 15), width=0.5, divergence=0.1),
                   dict(mesh=m, p1=(-5, 5), p2=(15, 5), width=0.5, divergence=0.1)]
        det = detector_array.detector_array(func_name='tomomak.detectors.detectors2d.detector2d', kwargs_list=kw_list)
        mod.detector_signal = [0, 0]
        mod.detector_geometry = det



