from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d, objects3d
from tomomak.util import geometry3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian, polar, toroidal
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import numpy as np

# # One of the main TOMOMAK features is the ability to work with non-Cartesian coordinates.
# # This tutorial explains how to use them correctly.
#
# # Let's start with 2D polar coordinate system. This system has two coordinates: radius and rotation angle.
# # In tomomak radius can be represented using 1D cartesian axis and rotation using 1D polar axis.
# # When TOMOMAK sees this axes combination, it automatically understands that we work with 2D polar system.
# axes = [cartesian.Axis1d(name="R", units="cm", size=20, upper_limit=10),
#         polar.Axis1d(name="phi", units="rad", size=15)]
# m = mesh.Mesh(axes)
# mod = model.Model(mesh=m)
#
# # Now let's what the circle will look like in this coordinates
# mod.solution = objects2d.ellipse(m, ax_len=(5.2, 5.2))
# mod.plot2d()
# # Well, looks fine, but how will it look in cartesian coordinates?
# # You don't need to use your imagination - just use cartesian_coordinates=True argument:
# mod.plot2d(cartesian_coordinates=True)
# # We see that there is a small artefact ring around the circle
# # due to the fact that this cells do not fully intersect with the circle, but otherwise everything looks fine.
# # Now let's look at the rectangle
# mod.solution = objects2d.rectangle(m,  size=(5, 5))
# mod.plot2d(cartesian_coordinates=True)
# # Well, it doesn't look like a rectangle at all. So important lesson is to choose correct mesh,
# # when you work with the limited data or on a grid with the large cells.
#
# # Now a few words about the detectors.
# # All the detector in the detectors2d module are positioned in the cartesian coordinates,
# # but you can work with them with any non-cartesian mesh.
# mod.detector_geometry = detectors2d.two_pi_detector_array(m, focus_point=(0, 0), radius=50, det_num=40)
# mod.plot2d(data_type='detector_geometry', cartesian_coordinates=True)
# mod.detector_geometry = detectors2d.parallel_detector(m, (-20, 0), (20, 0), width=1, number=10, shift=1)
# mod.plot2d(data_type='detector_geometry', cartesian_coordinates=True)
# # Note that detector_geometry plot shows intersection area of each cell with the detector line of sight,
# # so when the cell area is small, assigned cell value will be also small.

# Now let's switch to 3D.
# We will consider toroidal geometry, which is 2D polar geometry, rolled into o torus of a defined radius.
# In order to create mesh we should add 3rd toroidal axis.
axes = [toroidal.Axis1d(radius=20, name="theta", units="rad", size=25),
        polar.Axis1d(name="phi", units="rad", size=12),
        cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=5)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
# First af all we want to know what the cell of such a grid looks like. It will be a 2D bended polar cell.
# Let's look at the random cell.
geometry3d.show_cell(m, cell_index=(3, 3, 3))
# And this is what the center cell looks like.
geometry3d.show_cell(m, cell_index=(0, 0, 0))
# Now if you want to create an object, defined in the cartesian coordinates, you can use objects3d module.
mod.solution = objects3d.point_source(m, (0, 0, 0), index=(0, 1, 2))
mod.plot3d(cartesian_coordinates=True, axes=True)
#
# how can we create toroidally symmetric figure?


# And remember that you can easily add new coordinate systems.
# In order to do this you should create a new class that inherits 1D, 2D or 3D axis from abstract_axes module.
# After that you need to implement cell_edges2d_cartesian, cell_edges3d_cartesian,
# cartesian_coordinates, from_cartesian and, probably, create  a small adapter for plot_2d and plot_3d methods.
# See any non-cartesian axis for example - it is easier, than it sounds.