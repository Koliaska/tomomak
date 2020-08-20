from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.detectors import detectors3d, signal
from tomomak.iterators import ml
from tomomak.iterators import statistics


# This is a basic 3D tomography example.

# let's start by creating a 3D cartesian coordinate system.
# Since 3D tomography is much slower than 2D, in the example we will use 10x10x10 grid,
# but you can increase these values.
axes = [cartesian.Axis1d(name="X", units="cm", size=10, upper_limit=10),
        cartesian.Axis1d(name="Y", units="cm", size=10, upper_limit=10),
        cartesian.Axis1d(name="Z", units="cm", size=10, upper_limit=10)]
mesh = mesh.Mesh(axes)
mod = model.Model(mesh=mesh)
# Now let's create a test object. It will be a sphere-like source with 1/R^2 density.
# Note that several object types are supported, including arbitrary objects, defined by vertices and faces,
# however calculation of the most objects intersection with each grid cell will take significant time.
# This is the price to pay for the versatility of the framework.
real_solution = objects3d.point_source(mesh, (5, 5, 5))
# Let's plot the data in different ways.
mod.solution = real_solution
mod.plot2d()
mod.plot3d(axes=True, style=3)
# Open Mayavi pipeline -> Volumes -> CTF and try to change alpha channel curve (opacity) -> press update CTF.
# This way, for example, you can hide the outer layers to look inside the object.
mod.plot3d(axes=True, style=2)
# Open Mayavi pipeline -> IsoSurface -> Actor and try to change Point size or opacity.

# Ok, now let's add some detectors
mod.solution = None
# We will use an array of hermetic detectors.
# Note that several object types are supported,
# however calculation of the most detector line of sights intersection with each grid cell will take significant time.
# The good news is that you need to do these calculations only once for a given geometry.
# After that it is recommended to save calculation results, for example using >> mod.save("3dtomography.tmm")
# There are ways to accelerate the calculations using multiprocess (see GPU and CPU acceleration example)
# for example with detector_array function. Also detector line of sights may be represented as zero-width rays
# (line_detector function with parameter calc_volume = False), which is much faster, but usually not recommended.
mod.detector_geometry = detectors3d.four_pi_detector_array(mesh, focus_point=(5, 5, 5),
                                                           radius=10, theta_num=20, phi_num=20)

mod.plot3d(axes=True, style=3, data_type='detector_geometry')
# You can add other detectors - just uncomment the line and append the result  using mod.add_detector(det)
# det = [detectors3d.aperture_detector(mesh, [(4, -5, 4), (6, -5, 4), (4, -5, 6), (6, -5, 6)],
#                                      [(4, 0, 4), (6, 0, 4), (4, 0, 6), (6, 0, 6)])]
# det = [detectors3d.aperture_detector(mesh, [(4, -5, 4), (6, -5, 4), (4, -5, 6), (6, -5, 6)],
#                                                        [(4, 0, 4), (6, 0, 4), (4, 0, 6), (6, 0, 6)])]
# ver = np.array([[0, 0, 0], [0, 0, 10], [0, 10, 0], [10, 0, 0]])
# det = [detectors3d.custom_detector(mesh, ver, radius_dependence=False)]
# det = [detectors3d.line_detector(mesh, (5, 5, 12), (10, 2, 0), 1, calc_volume=True)]
# det = [detectors3d.cone_detector(mesh, (5, 5, 20), (5, 5, 0), 0.3)]
# det = [detectors3d.line_detector(mesh, (5, 5, 12), (10, 2, 0), None, calc_volume=False)]
# det = detectors3d.fan_detector(mesh, (-5, 5, 5), [[5, 2, 6], [5, 7, 6], [5, 2, 4], [5, 7, 4]],
#                                number=[4, 4], divergence=0.3)
det_signal = signal.get_signal(real_solution, mod.detector_geometry)
mod.detector_signal = det_signal
# Now let's find the solution
solver = Solver()
solver.statistics = [statistics.RN(), statistics.RMS()]
solver.real_solution = real_solution
solver.iterator = ml.ML()
steps = 1500
solver.solve(mod, steps=steps)
# Now we can see the solution.
mod.plot3d(axes=True, style=3)
mod.plot2d()
solver.plot_statistics()
# Of course, the result we got is not ideal due to limitations of our synthetic experiment.
# Luckily there are ways to improve our calculations, described in the more advanced examples.
