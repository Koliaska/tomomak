from tomomak import model
from tomomak.test_objects import objects2d, objects3d
from tomomak.util import geometry3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian, polar, toroidal
from tomomak.detectors import detectors2d, detectors3d
import numpy as np

# One of the main TOMOMAK features is the ability to work with non-Cartesian coordinates.
# Non-cartesian coordinates may be used only for spatial axes.
# Non-spatial axes may be irregular, however they should be orthogonal to other axes,
# so cartesian.Axis1d class should be used.
# This tutorial explains how to work with non-Cartesian coordinates correctly.

# Let's start with 2D polar coordinate system. This system has two coordinates: rotation angle and radius.
# In Tomomak radius can be represented using 1D cartesian axis and rotation - using 1D polar axis.
# When TOMOMAK sees this axes combination, it automatically understands that we work with 2D polar system.
axes = [polar.Axis1d(name="phi", units="rad", size=15),
        cartesian.Axis1d(name="R", units="cm", size=20, upper_limit=10)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)

# Now let's see what the circle will look like in this coordinate system
mod.solution = objects2d.ellipse(m, ax_len=(5.2, 5.2))
mod.plot2d()
# Well, looks fine, but how it will look in cartesian coordinates?
# You don't need to use your imagination - just use cartesian_coordinates=True argument:
mod.plot2d(cartesian_coordinates=True)
# We see that there is a small artefact ring around the circle due to the fact,
# that this cells does not fully intersect with out object, but everything else looks fine.
# Now let's look at the rectangle
mod.solution = objects2d.rectangle(m,  size=(5, 5))
mod.plot2d(cartesian_coordinates=True)
# Well, it doesn't look like a rectangle at all. So important lesson is to choose correct mesh,
# when you work with the limited data or with a grid that has large cells.

# Now a few words about the detectors.
# All the detector in the detectors2d / detector3d module are positioned in the cartesian coordinates,
# but you can work with them on any non-cartesian mesh.
mod.detector_geometry = detectors2d.two_pi_detector_array(m, focus_point=(0, 0), radius=50, det_num=40)
mod.plot2d(data_type='detector_geometry', cartesian_coordinates=True)
mod.detector_geometry = detectors2d.parallel_detector(m, (-20, 0), (20, 0), width=1, number=10, shift=1,
                                                      radius_dependence=False)
mod.plot2d(data_type='detector_geometry', cartesian_coordinates=True)
# Note that detector_geometry plot shows intersection area of each cell with the detector line of sight,
# so when the cell area is small, assigned cell value will be also small.
# If you want to see normalised area, that is more intuitive,  use data_type='detector_geometry_n'
mod.plot2d(data_type='detector_geometry_n', cartesian_coordinates=True)

# Now let's switch to 3D.
# We will consider toroidal geometry, which is 2D polar geometry, rolled into o torus of a defined radius.
# In order to create mesh we should add 3rd toroidal axis.
# Let's use only half of the toroidal grid, so we can see the crossection:
axes = [toroidal.Axis1d(radius=20, name="theta", units="rad", size=15, upper_limit=np.pi),
        polar.Axis1d(name="phi", units="rad", size=12),
        cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=10)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
# First of all we want to know what the cell of such a grid looks like. It will be a 2D bended polar cell.
# Let's look at the random cell.
geometry3d.show_cell(m, cell_index=(3, 3, 3))
# And this is what the center cell, closest to R = 0 looks like.
geometry3d.show_cell(m, cell_index=(0, 0, 0))
# Now if you want to create an object, defined in the cartesian coordinates, you can use objects3d module.
mod.solution = objects3d.point_source(m, (0, 0, 0), index=(0, 1, 2))
mod.plot3d(cartesian_coordinates=True, axes=True)
# But you can also work directly with numpy arrays inside of the mesh.
# E.g. let's create toroidally symmetric distribution with the 2D rectangle distribution in the crossection.
mod.solution = objects2d.rectangle(m, center=(0, 0), size=(8, 8), index=(1, 2), broadcast=True)
mod.plot3d(cartesian_coordinates=True, axes=True)
# Well. it looks a bit round, but taht's what we expect from the polar coordinate system.
# You can see, that there is gradient glow around the main shape - that's how 3D render works.
# It assumes that physical distribution doesn't have sharp edges.

# Detectors work similar to objects.
# Note, that 3D interpolation for render will take significant time.
mod.detector_geometry = detectors3d.four_pi_detector_array(m, focus_point=(0, 0, 0), radius=45, theta_num=2, phi_num=3)
mod.plot3d(data_type='detector_geometry', cartesian_coordinates=True, axes=True, style=1)

# After you've created a model you can work with it as usual.

# And remember that you can easily add new coordinate systems.
# In order to do this you should create a new class that inherits 1D, 2D or 3D axis from abstract_axes module.
# After that you need to implement cell_edges2d_cartesian, cell_edges3d_cartesian,
# cartesian_coordinates, from_cartesian and, probably, create  a small adapter for plot_2d and plot_3d methods.
# For example see any non-cartesian axis - it is easier, than it sounds.
