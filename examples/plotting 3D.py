import numpy as np
import tomomak as tm
from tomomak.mesh import cartesian, polar, toroidal
from tomomak.test_objects import objects2d

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
noise = np.random.normal(0, 0.01, real_solution.shape)
mod.solution = real_solution + noise
# Now it's time to do plotting
mod.plot3d(style=1)
# You can play with the Mayavi pipeline (upper left corner) in order to get better-looking figure.
# For example try to change Data minimum in contours tab of the isoSurface, or opacity in Actor tab.
# There are other predefined styles which you can use as a starting position for plot generation.
# Also let's add coordinate axes and change the style.
mod.plot3d(style=2, axes=True)
# Try to change Point size in Actor tab.
# Another style:
mod.plot3d(style=3, axes=True)
# Open Mayavi pipeline -> Volumes -> CTF and try to change alpha channel curve (opacity) -> press update CTF.
# This way, for example, you can hide the outer layers to look inside the object.

# Finally, a voxel plot, which shows the exact data in the cells.
# Shape of each cell is taken into account.
mod.plot3d(style=0)
# Use Min_val and Max_val sliders to hide unnecessary data.
# And remember, that it is still possible to make voxels transparent:
# Open Mayavi pipeline -> Surface -> Property -> Opacity.

# You can also plot detector geometry this way.
# Let's create test geometry. You can switch between detectors with the bottom slider.
# In this example we will use same norm for all detectors.
mod.detector_geometry = np.array([mod.solution, mod.solution + noise, mod.solution + noise * 2])
mod.plot3d(data_type='detector_geometry', equal_norm=True, style=3)

# Another thing you should know - is how to speed up voxel plot rendering when working in non-cartesian coordinates.
# Non-cartesian axes has members RESOLUTION3D or RESOLUTION2D which may be changed
# (but only one time, before the first calculations ate performed):
axes = [toroidal.Axis1d(radius=15, name="theta", units="rad", size=15, upper_limit=np.pi),
        polar.Axis1d(name="phi", units="rad", size=12),
        cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=10)]
axes[0].RESOLUTION3D = 3
axes[1].RESOLUTION2D = 3
mesh = tm.Mesh(axes)
mod = tm.Model(mesh=mesh)
real_solution = objects2d.ellipse(mesh, ax_len=(5.2, 5.2), index=(1, 2))
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution + noise
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)

# And that's how you deal with basic 3D plotting.
