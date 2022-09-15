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

mod.solution = real_solution
mod.plot2d()

mod.plot1d(index=0)

mod.solution = None

det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=15,
                                     line_num=22,
                                     width=1,
                                     divergence=0.2)

det_signal = signal.get_signal(real_solution, det)
mod.detector_signal = det_signal
mod.detector_geometry = det

mod.plot2d(data_type='detector_geometry')

print(mod)


pipe = pipeline.Pipeline(mod)
r = rescale.Rescale((20, 20))
pipe.add_transform(r)

mod.solution = real_solution
pipe.forward()
real_solution = mod.solution
mod.plot2d()
mod.solution = None

solver = Solver()

solver.statistics = [statistics.RN(), statistics.RMS()]

solver.real_solution = real_solution

solver.iterator = ml.ML()

steps = 50
solver.solve(mod, steps=steps)
mod.plot2d()
solver.plot_statistics()


solver.iterator = algebraic.ART()

solver.constraints = [tomomak.constraints.basic.Positive()]

solver.stop_conditions = [statistics.RMS()]
solver.stop_values = [15]

steps = 10000

solver.iterator.alpha = np.linspace(0.1, 0.01, steps)
solver.solve(mod, steps=steps)
mod.plot2d()
solver.plot_statistics()

mod.save("basic_func_model.tmm")

