from tomomak.mesh.mesh import *
from tomomak import model
from tomomak.test_objects import objects2d, objects3d
from tomomak.mesh import mesh
from tomomak.mesh import polar, toroidal, level
import numpy as np
import tomomak.util.eqdsk as eqdsk
from tomomak import util


# This is an example, showing how to use this framework with the tokamak-specific geometry.


# For tomography in tokamak cartesian or cylindrical coordinates may be used.
# However it is much more convenient to use common tokamak coordinate system (psi, theta, phi),
# where psi is poloidal magnetic flux, determining radial position
# (normalized magnetic coordinate rho may be used instead),
# theta is poloidal angle and phi is toroidal angle. In this case it is easier to set penalties_and_constraints
# during the reconstruction, e.g. radial or poloidal smoothness.

# The first step is to get 2D magnetic flux. Routine for the reading of the EFIT files already exists.
g = eqdsk.read_eqdsk('globus.g', b_ccw=-1)
# In order to simplify data let's add constant, so central psi value is 0
g = eqdsk.norm_psi(g)
# after this step we may create mask for the space outside of the plasma using geometry2d.in_out_mask(),
# however in our case it is not required.

# Since sophisticated poloidal coordinates, such as equal volumes or straight field lines require psi but not rho,
# we will use (psi, theta, phi) coordinates instead of (rho, theta, phi). For other coordinate systems
# (rho, theta, phi) may be used. However it is easy to convert rho to psi:
# let's do this for the irregular grid in rho coordinates

psi_edges = eqdsk.rho_to_psi(g, np.array([0.0, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.999]))

# Now we can create a 3D coordinate system. For phi we will use "toroidal" coordinates,
# for theta - "poloidal and for psi - "level" coordinates.
# level coordinates is used to define the "height" map of some quantity. Combination of level and poloidal
# is used for the map, where the highest/lowest is in the center, and the level monotonically decreases/increases
# when moving away from the center. For example, it may be used to map the mountain (without gendarmes),
# or, as in our case, to map tokamak equilibrium.
# There are several ways to choose a polar angle. Here we will choose the case, when volumes
# of all poloidal elements are the same. You can try to use 'eq_angle', 'eq_arc' or 'straight_line' instead.
# For better view we will analyze only 3pi/2 angle in toroidal direction.

axes = [toroidal.Axis1d(radius=0.0, name="phi", units="rad", size=10, upper_limit=3*np.pi/2),
        level.Axis1d(level_map=g['psi'], x=g['r'], y=g['z'], x_axis=g['raxis'], y_axis=g['zaxis'],
                     name="rho", units="a.u.", cart_units='m', edges=psi_edges, polar_angle_type='eq_vol'),
        polar.Axis1d(name="theta", units="rad", size=16)]

# It is possible to change number of points in each direction per one element:
axes[0].RESOLUTION3D = 4
axes[2].RESOLUTION2D = 5
# level resolution also may be changed however, be careful.
axes[1].resolution2d = 8

# Now we can create model
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)

# When the geometry is created you can work with it like with any other geometry.

# Now let's analyze some examples.
# First of all, let's see what the grid looks like.
# The first step is to create a uniform distribution in the poloidal cross section.
# The easiest way to do this is to calculate the intersection with a very big 2D object.
res = objects2d.ellipse(m, ax_len=(2, 2), index=(1, 2), center=(0.36, 0), broadcast=True)
mod.solution = res
mod.plot2d(style='colormesh', cartesian_coordinates=True, index=(1, 2))

# The small difference (look at the colorbar) between the cells is due to numerical error.

# Since we created this distribution in 2D, it will not be uniform in 3D.
# In order to create uniform 3D distribution in cartesian coordinates we should use intersection with 3D object
# or do the following thing:

areas = util.geometry2d.cell_areas(m, (1, 2))
vols = util.geometry3d_trimesh.cell_volumes(m)
res = res * vols / areas
mod.solution = res
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)

# Finally let's create (4,1) rotational mode at rho=0.5 magnetic surface (psi_edges[3]) and look at the visualisation
# For this case we better create a mesh with more cells.
axes = [toroidal.Axis1d(radius=0.0, name="phi", units="rad", size=40, upper_limit=3*np.pi/2),
        level.Axis1d(level_map=g['psi'], x=g['r'], y=g['z'], x_axis=g['raxis'], y_axis=g['zaxis'],
                     name="rho", units="a.u.", cart_units='m', edges=psi_edges, polar_angle_type='eq_angle'),
        polar.Axis1d(name="theta", units="rad", size=40)]
# It is possible to change number of points in each direction per one element:
axes[0].RESOLUTION3D = 3
axes[2].RESOLUTION2D = 3

m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
res = np.abs(objects3d.rotational_mode(m, 4, 2, psi_edges[3], (0, 2, 1), shift=1.2))
mod.solution = res
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)

# Use bottom slider to hide all unimportant elements.
