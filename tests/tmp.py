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


# This is an example of a basic framework functionality.
# You will learn how to use framework, steps you need to follow in order to get the solution.
# More advanced features are described in advanced examples.

# The first step is to create coordinate system. We will consider 2D cartesian coordinates.
# Let's create coordinate mesh. First axis will be from 0 to 10 cm and consist of 20 segments.
# Second - from 0 to 10 cm of 30 segments.
# In this case the solution will be described by the 20x30 array.
axes = [cartesian.Axis1d(name="X", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Y", units="cm", size=30, upper_limit=10)]
mesh = mesh.Mesh(axes)
# Now we can create Model.
# Model is one of the basic tomomak structures which stores information about geometry, solution and detectors.
# At present we only have information about the geometry.
mod = model.Model(mesh=mesh)

det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=5,
                                     line_num=4,
                                     width=1,
                                     divergence=0.2)
# Now we can calculate signal of each detector.
# Of course in the real experiment you measure detector signals so you don't need this function.


# Let's take a look at the detectors geometry:
mod.detector_geometry = det
mod.plot2d(data_type='detector_geometry', fig_name='Detector geometry')
mod.plot1d(data_type='detector_geometry', fig_name='Detector geometry')


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



