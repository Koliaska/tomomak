from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import numpy as np

axes = [cartesian.Axis1d(name="X", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Y", units="cm", size=30, upper_limit=10)]
mesh = mesh.Mesh(axes)

mod = model.Model(mesh=mesh)

real_solution = objects2d.polygon(mesh, [(1, 1), (4, 8), (7, 2)])


det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=14,
                                     line_num=20,
                                     width=1,
                                     divergence=0.2)

det_signal = signal.get_signal(real_solution, det)
print(det_signal)
noisy_det_signal = signal.add_noise(det_signal, 10)
mod.detector_signal = noisy_det_signal
mod.detector_geometry = det

#os.environ["TM_GPU"] = "1"

steps = 100
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = algebraic.ART()
solver.constraints = [tomomak.constraints.basic.Positive()]
solver.solve(mod, steps=steps)
mod.plot2d(fig_name='No reg')

mod.solution = None
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = algebraic.ART()
solver.constraints = [tomomak.constraints.regression.Entropy(alpha=0.1),
                      tomomak.constraints.basic.Positive()]
solver.solve(mod, steps=steps)
mod.plot2d(fig_name='entropy')

mod.solution = None
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = algebraic.ART()
solver.constraints = [tomomak.constraints.regression.Lasso(alpha=0.1),
                      tomomak.constraints.basic.Positive()]
solver.solve(mod, steps=steps)
mod.plot2d(fig_name='Lasso')

mod.solution = None
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = algebraic.ART()
solver.constraints = [tomomak.constraints.regression.Tikhonov(alpha=0.5),
                      tomomak.constraints.basic.Positive()]

solver.solve(mod, steps=steps)
mod.plot2d(fig_name='Tikhonov')
import scipy.ndimage
func = scipy.ndimage.gaussian_filter1d
mod.solution = None
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = algebraic.ART()
solver.constraints = [tomomak.constraints.basic.Positive(),
                      tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=0.1, sigma=1),
                      tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=0.1, sigma=1)
                      ]
solver.solve(mod, steps=steps)
mod.plot2d(fig_name='Smooth')

mod.solution = real_solution
mod.plot2d(fig_name='Test object')



