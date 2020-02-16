import numpy as np
import tomomak as tm
from tomomak.mesh import cartesian

# This example shows how to utilize 3D plotting.
# Generally plotting a 3D distribution may be pretty challenging.
# Probably you will need to do a lot of fine-tuning to get a good-looking figure.
# Therefore, a versatile plotting system is provided with the framework.

# Let's create our mesh
axes = [cartesian.Axis1d(name="X", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Y", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Z", units="cm", size=20, upper_limit=10)]
mesh = tm.Mesh(axes)
mod = tm.Model(mesh=mesh)

# The next step - is to create a synthetic object.
# Let's create 2D circle. Since our mesh is 3D it will be automatically broadcasted to 3rd dimension,
# so we get a cylinder.
real_solution = tm.test_objects.objects2d.ellipse(mesh, (5, 5), (3, 3))
# Now let's add some noise to this object in order to simulate realistic distribution.
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution + noise
# Now it's time to do plotting
mod.plot3d()
# You can play with the Mayavi pipeline (upper left corner) in order to get better-looking figure.
# For example try to change Data minimum in contours tab of the isoSurface, or opacity in Actor tab.
# There are other predefined styles which you can use as a starting position for plot generation.
mod.plot3d(style=2)
# Try to change Point size in Actor tab.

# You can also plot detector geometry this way.
# Let's create test geometry. You can switch between detectors with the bottom slider.
# In this example we will use same norm for all detectors.
mod.detector_geometry = np.array([mod.solution, mod.solution + noise, mod.solution + noise * 2])
mod.plot3d(data_type='detector_geometry', equal_norm=True)
# And that's how you deal with basic 3D plotting.