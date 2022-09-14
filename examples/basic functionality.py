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
import tomomak.penalties_and_constraints.basic
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
# Now let's create synthetic 2D object to study.
# We will consider triangle.
real_solution = objects2d.polygon(mesh, [(1, 1), (4, 8), (7, 2)])
# Model.solution is the solution we are looking for.
# It will be obtained at the end of this example.
# However, if you already know supposed solution (for example you get it with simulation),
# you can use it as first approximation by setting Model.solution = *supposed solution*.
# Recently we've generated test object, which is, of course, real solution.
# A trick to visualize this object is to temporarily use it as model solution.
mod.solution = real_solution
mod.plot2d()
# You can also make 1D plot. In this case data will be integrated over 2nd axis.
mod.plot1d(index=0)
# After we've visualized our test object, it's time to set model solution to None and try to find this solution fairly.
mod.solution = None

# Next step is to provide information about the detectors.
# Let's create 15 fans with 22 detectors around the investigated object.
# The fans will be positioned around the focus point (5, 5) at a distance of 11 cm.
# Each line will have 1 cm width and 0.2 Rad divergence.
# Note that number of detectors = 330 < solution cells = 600, so it's impossible to get perfect solution.
det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=15,
                                     line_num=22,
                                     width=1,
                                     divergence=0.2)
# Now we can calculate signal of each detector.
# Of course in the real experiment you measure detector signals so you don't need this function.
det_signal = signal.get_signal(real_solution, det)
mod.detector_signal = det_signal
mod.detector_geometry = det
# Let's take a look at the detectors geometry:
mod.plot2d(data_type='detector_geometry')
# It's also possible to get short model summary by converting the model object to string.
print(mod)

# The next step is optional. You can perform transformation with existing geometry,
# e.g. switch to another coordinate system or to basic function space.
# Convenient way to do this is to use pipeline. Once pipeline is created,
# you can do transformations using forward() method.
# And if you wish to perform backward transformation later, you can use backward() method.
# let's rescale our model to 20x20 cells. Rescale class performs transformation to the
# keeps axes types but changes number of segments. If the axis is irregular, rescaling take it into account.
pipe = pipeline.Pipeline(mod)
r = rescale.Rescale((20, 20))
pipe.add_transform(r)
# Our real solution should also be changed, so we need to use same trick again.
mod.solution = real_solution
pipe.forward()
real_solution = mod.solution
print(f"Rescaled model:\n{mod}")
mod.plot2d()
mod.solution = None
# The rescaling is successful.
# If you want to switch to previous 20x30 cells case just use pipe.backward().
# Now let's find the solution. In order to do so we need to create solver.
solver = Solver()
# We can easily track different statistics. Let's track residual norm and Chi^2 statistics.
# When the calculations are over you can plot the results or find statistical value at each step
# in "data" attribute of each object, e.g. in solver.statistics[0].data
solver.statistics = [statistics.RN(), statistics.RMS()]
# RMS need to know real solution in order to perform calculations.
solver.real_solution = real_solution

# Finally, let's choose method, we would like to use in order to find the solution.
# We start with  maximum likelihood method.
solver.iterator = ml.ML()
# Let's do 50 steps and see resulted image and statistics.
steps = 50
solver.solve(mod, steps=steps)
mod.plot2d()
solver.plot_statistics()

# Now let's change to  algebraic reconstruction technique.
solver.iterator = algebraic.ART()
# We can also add some penalties_and_constraints. This is important in the case of limited date reconstruction.
# For now let's assume that all values are positive. Note that ML method didn't need this constraint,
# since one of it's features is to preserve solution sign.
solver.constraints = [tomomak.constraints.basic.Positive()]
# It's possible to choose early stopping criteria for our reconstruction.
# In this example we want residual mean square error to be < 15 %.
# In the real world scenario you will not know real solution,
# so you will use other stopping criterias, e.g. residual norm.
solver.stop_conditions = [statistics.RMS()]
solver.stop_values = [15]
# Also we should limit number of steps in the case it's impossible to reach such accuracy.
steps = 10000
# Finally, let's make decreasing step size. It will start from 0.1 and decrease a bit at every step.
solver.iterator.alpha = np.linspace(0.1, 0.01, steps)
# And here we go:
solver.solve(mod, steps=steps)
mod.plot2d()
solver.plot_statistics()

# Finally we can save our model for later usage.
mod.save("basic_func_model.tmm")

# And that's it. You get solution, which is, of course, not perfect,
# but world is not easy when you work with the limited data.
# There are number of ways to improve your solution, which will be described in other examples.
