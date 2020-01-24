from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak import iterators
from tomomak.iterators import ml, algebraic#, gpu
from tomomak.iterators import statistics
import tomomak.constraints.basic
from mpl_toolkits.mplot3d import Axes3D
import itertools


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

# conda remove vtk conda remove mayavi and installing with pip install vtk and pip install mayavi. Thanks for your help!




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



# This is an example of a basic framework functionality.
# You will learn how to use framework, steps you need to follow in order to get the solution.
# More advanced features are described in advanced examples.

# The first step is to create coordinate system. We will consider 2D cartesian coordinates.
# Let's create coordinate mesh. First axis will consist of 20 segments. Second - of 30 segments.
# This means that solution will be described by the 20x30 array.
axes = [cartesian.Axis1d(name="X", units="cm",  coordinates=(0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5, 3,3.5, 4,4.5, 5, 7, 9,12)),
        cartesian.Axis1d(name="Y", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Z", units="cm", size=20, upper_limit=10)]
mesh = mesh.Mesh(axes)
mod = model.Model(mesh=mesh)
real_solution = objects2d.ellipse(mesh, (5,5),(3,3))
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution + noise

from mayavi import mlab
x, y, z = np.mgrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x*y*z)/(x*y*z)

# mlab.pipeline.volume(mlab.pipeline.scalar_field(x,y,z,s), vmin=0, vmax=0.8)
#
# mlab.show()
mod.detector_geometry = np.array([mod.solution, mod.solution + noise, mod.solution + noise * 2])
obj, cb = mod.plot3d(data_type = 'detector_geometry', equal_norm=True, axes=True)
mod.plot2d()



# Create the data.

# from mayavi import mlab
#
# def test_mesh():
#     """A very pretty picture of spherical harmonics translated from
#     the octaviz example."""
#     pi = np.pi
#     cos = np.cos
#     sin = np.sin
#     dphi, dtheta = pi / 250.0, pi / 250.0
#     [phi, theta] = np.mgrid[0:pi + dphi * 1.5:dphi,
#                             0:2 * pi + dtheta * 1.5:dtheta]
#     m0 = 4
#     m1 = 3
#     m2 = 2
#     m3 = 3
#     m4 = 6
#     m5 = 2
#     m6 = 6
#     m7 = 4
#     r = sin(m0 * phi) ** m1 + cos(m2 * phi) ** m3 + \
#         sin(m4 * theta) ** m5 + cos(m6 * theta) ** m7
#     x = r * sin(phi) * cos(theta)
#     y = r * cos(phi)
#     z = r * sin(phi) * sin(theta)
#
#     return mlab.mesh(x, y, z, colormap="bone")
#
# def test_mesh2():
#     """A very pretty picture of spherical harmonics translated from
#     the octaviz example."""
#     pi = np.pi
#     cos = np.cos
#     sin = np.sin
#     dphi, dtheta = pi / 250.0, pi / 250.0
#     [phi, theta] = np.mgrid[0:pi + dphi * 1.5:dphi,
#                             0:2 * pi + dtheta * 1.5:dtheta]
#     m0 = 4
#     m1 = 3
#     m2 = 2
#     m3 = 3
#     m4 = 6
#     m5 = 2
#     m6 = 6
#     m7 = 4
#     r = sin(m0 * phi) ** m1 + cos(m2 * phi) ** m3 + \
#         sin(m4 * theta) ** m5 + cos(m6 * theta) ** m7
#     x = r * sin(phi) * cos(theta) + 10
#     y = r * cos(phi)
#     z = r * sin(phi) * sin(theta)
#
#     return mlab.mesh(x, y, z, color = (0.1, 1, 0.2))
# mlab.figure(figure="Figfff", bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 350))
# test_mesh()
# test_mesh2()
# mlab.show()

#
# # This is an example of a basic framework functionality.
# # You will learn how to use framework, steps you need to follow in order to get the solution.
# # More advanced features are described in advanced examples.
#
# # The first step is to create coordinate system. We will consider 2D cartesian coordinates.
# # Let's create coordinate mesh. First axis will consist of 20 segments. Second - of 30 segments.
# # This means that solution will be described by the 20x30 array.
# axes = [cartesian.Axis1d(name="X", units="cm", size=20, upper_limit=10),
#         cartesian.Axis1d(name="Y", units="cm", size=20, upper_limit=10),
#         cartesian.Axis1d(name="Y", units="cm", size=20, upper_limit=10)]
# mesh = mesh.Mesh(axes)
# mod = model.Model(mesh=mesh)
# real_solution = objects2d.ellipse(mesh, (5,5),(3,3))
# mod.solution = real_solution
# mod.plot3d()
# # Now we can create Model.
#
# # Model is one of the basic tomomak structures which stores information about geometry, solution and detectors.
# # At present we only have information about the geometry.
# mod = model.Model(mesh=mesh)
# # Now let's create synthetic 2D object to study.
# # We will consider triangle.
# real_solution = objects2d.ellipse(mesh, (5,5),(3,3))
# # Model.solution is the solution we are looking for.
# # It will be obtained at the end of this example.
# # However, if you already know supposed solution (for example you get it with simulation),
# # you can use it as first approximation by setting Model.solution = *supposed solution*.
# # Recently we've generated test object, which is, of course, real solution.
# # A trick to visualize this object is to temporarily use it as model solution.
# mod.solution = real_solution
# mod.plot2d()
# # You can also make 1D plot. In this case data will be integrated over 2nd axis.
# mod.plot1d(index=0)
# # After we've visualized our test object, it's time to set model solution to None and try to find this solution fairly.
# mod.solution = None
#
# # Next step is to provide information about the detectors.
# # Let's create 15 fans with 22 detectors around the investigated object.
# # Each line will have 1 cm width and 0.2 Rad divergence.
# # Note that number of detectors = 330 < solution cells = 600, so it's impossible to get perfect solution.
# det = detectors2d.fan_detector_array(mesh=mesh,
#                                      focus_point=(5, 5),
#                                      radius=11,
#                                      fan_num=15,
#                                      line_num=22,
#                                      width=1,
#                                      divergence=0.2)
# # Now we can calculate signal of each detector.
# # Of course in the real experiment you measure detector signals so you don't need this function.
# det_signal = signal.get_signal(real_solution, det)
# mod.detector_signal = det_signal
# mod.detector_geometry = det
# # Let's take a look at the detectors geometry:
# mod.plot2d(data_type='detector_geometry')
# # It's also possible to get short model summary by converting the model object to string.
# print(mod)
#
# # The next step is optional. You can perform transformation with existing geometry,
# # e.g. switch to another coordinate system or to basic function space.
# # Convenient way to do this is to use pipeline. Once pipeline is created,
# # you can do transformations using forward() method.
# # And if you wish to perform backward transformation later, you can use backward() method.
# # let's rescale our model to 20x20 cells. Rescale class performs transformation to the
# # keeps axes types but changes number of segments. If the axis is irregular, rescaling take it into account.
# pipe = pipeline.Pipeline(mod)
# r = rescale.Rescale((20, 20))
# pipe.add_transform(r)
# # Our real solution should also be changed, so we need to use same trick again.
# mod.solution = real_solution
# pipe.forward()
# real_solution = mod.solution
# mod.plot2d()
# mod.solution = None
# # The rescaling is successful.
# # If you want to switch to previous 20x30 cells case just use pipe.backward(). Note that you will lost
#
# # Now let's find the solution. In order to do so we need to create solver.
# solver = Solver()
# # We can easily track different statistics. Let's track residual norm and Chi^2 statistics.
# # When the calculations are over you can plot the results or find statistical value at each step
# # in "data" attribute of each object, e.g. in solver.statistics[0].data
# solver.statistics = [statistics.RN(), statistics.RMS()]
# # RMS need to know real solution in order to perform calculations.
# solver.real_solution = real_solution
#
# # Finally, let's choose method, we would like to use in order to find the solution.
# # We start with  maximum likelihood method.
# solver.iterator = ml.ML()
# # Let's do 50 steps and see resulted image and statistics.
# steps = 50
# solver.solve(mod, steps=steps)
# mod.plot2d()
# solver.plot_statistics()
#
# # Now let's change to  algebraic reconstruction technique.
# solver.iterator = algebraic.ART()
# # We can also add some constraints. This is important in the case of limited date reconstruction.
# # For now let's assume that all values are positive. Note that ML method didn't need this constraint,
# # since one of it's features is to preserve solution sign.
# solver.constraints = [tomomak.constraints.basic.Positive()]
# # It's possible to choose early stopping criteria for our reconstruction.
# # In this example we want residual mean square error to be < 15 %.
# # In the real world scenario you will not know real solution,
# # so you will use other stopping criterias, e.g. residual norm.
# solver.stop_conditions = [statistics.RMS()]
# solver.stop_values = [15]
# # Also we should limit number of steps in the case it's impossible to reach such accuracy.
# steps = 10000
# # Finally, let's make decreasing step size. It will start from 0.1 and decrease a bit at every step.
# solver.iterator.alpha = np.linspace(0.1, 0.01, steps)
# # And here we go:
# solver.solve(mod, steps=steps)
# mod.plot2d()
# solver.plot_statistics()
#
# # And that's it. You get solution, which is, of course, not perfect,
# # but world is not easy when you work with the limited data.
# # There are number of ways to improve your solution, which will be described in other examples.
#
#
# axes = [Axis1d(name="X", units="cm", size=15, upper_limit=10),
#         Axis1d(name="Y", units="cm", size=15, upper_limit=10)]
# mesh = Mesh(axes)
# # Now we can create Model.
# # Model is one of the basic tomomak structures which stores information about geometry, solution and detectors.
# # At present we only have information about the geometry.
# mod = Model(mesh=mesh)
# # Now let's create synthetic 2D object to study.
# # We will consider triangle.
# real_solution = polygon(mesh, [(1, 1), (4, 8), (7, 2)])
# # Model.solution is the solution we are looking for.
# # It will be obtained at the end of this example.
# # However, if you already know supposed solution (for example you get it with simulation),
# # you can use it as first approximation by setting Model.solution = *supposed solution*.
# # Recently we've generated test object, which is, of course, real solution.
# # A trick to visualize this object is to temporarily use it as model solution.
# mod.solution = real_solution
#
# # After we've visualized our test object, it's time to set model solution to None and try to find this solution fairly.
# mod.solution = None
#
# # Next step is to provide information about the detectors.
# # Let's create 15 fans with 22 detectors around the investigated object.
# # Each line will have 1 cm width and 0.2 Rad divergence.
# # Note that number of detectors = 330 < solution cells = 600, so it's impossible to get perfect solution.
# det = detectors2d.fan_detector_array(mesh=mesh,
#                                      focus_point=(5, 5),
#                                      radius=11,
#                                      fan_num=1,
#                                      line_num=22,
#                                      width=1,
#                                      divergence=0.2)
# # Now we can calculate signal of each detector.
# # Of course in the real experiment you measure detector signals so you don't need this function.
# det_signal = signal.get_signal(real_solution, det)
# mod.detector_signal = det_signal
# mod.detector_geometry = det
# # Let's take a look at the detectors geometry:
# mod.plot2d(data_type='detector_geometry', equal_norm=True)
# mod.plot1d(data_type='detector_geometry',equal_norm=True )
# #axes = [Axis1d(name="x", units="cm", size=20), Axis1d(name="Y", units="cm", size=30), Axis1d(name="Y", units="cm", size=130)]
# axes = [Axis1d(name="x", units="cm", size=5), Axis1d(name="Y", units="cm", size=10), Axis1d(name="Z", units="cm", size=20)]
# #axes = [Axis1d(name="x", units="cm", size=21), Axis1d(name="Y", units="cm", coordinates=np.array([1, 3, 5, 7, 9, 13]),  lower_limit=0), Axis1d(name="z", units="cm", size=3)]
#
#
# # inters = axes[0].cell_edges3d(axes[1], axes[2])
# # rect = inters[0][0][0]
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# #
# # for p in rect:
# #     xs = p[0]
# #     ys = p[1]
# #     zs = p[2]
# #     ax.scatter(xs, ys, zs)
# # plt.show()
#
# mesh = Mesh(axes)
#
# solution = polygon(mesh, [(1,1), (1, 8), (7, 9), (7, 2)])
#
# #solution = detectors2d.line2d(mesh, (-1, 7), (11, 3), 1, divergence=0.1, )
# det = detectors2d.fan_detector_array(mesh, (5,5), 11, 10, 22, 1, incline=0 )
#
# det_signal = signal.get_signal(solution, det)
# #det = detectors2d.parallel_detector(mesh,(-10, 7), (11, 3), 1, 10, 0.2)
#
# #det = detectors2d.fan_detector(mesh, (-3, 7), (11, 7), 0.5, 10, angle=np.pi/2)
# # solution = rectangle(mesh,center=(6, 4), size = (4, 2.7), index = (1,2))
# # solution  = real_solution(mesh)
# # solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
# # solution  = cone(mesh,center=(5, 5), ax_len=(3, 7))
#
# # detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
# #detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# # detector_signal=np.array([3, 1, 5, 4])
# # solution = np.array([[1, 2, 1], [5, 5.1, 4]])
# #solution= sparse.COO(solution)
#
# mod = Model(mesh=mesh,  detector_signal = det_signal, detector_geometry=det, solution = solution)
# mod.plot3d()
# #mod.plot2d(index=(0,1))
# mod.solution = None
# solver = Solver()
# steps = 100
# solver.real_solution = solution
# import cupy as cp
# solver.iterator = ml.ML()
# # solver.alpha = cp.linspace(1, 1, steps)
# #solver.iterator = gpu.MLCuda()
# #solver.iterator.alpha = cp.linspace(1, 1, steps)
# solver.statistics = [statistics.rms]
# # solver.alpha = np.linspace(1, 1, steps)
# #solver.iterator = algebraic.ART()
# #solver.iterator = algebraic.SIRT(n_slices=3, iter_type='SIRT')
# solver.iterator.alpha =  np.linspace(0.1, 0.0001, steps)
#
# import scipy.ndimage
# func = scipy.ndimage.gaussian_filter1d
# #c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=1, sigma=2)
# c2 = tomomak.constraints.basic.ApplyFunction(scipy.ndimage.gaussian_filter, sigma=1, alpha=1)
# # c3 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=1, sigma=2)
# solver.constraints = [tomomak.constraints.basic.Positive(), c2]
# import time
# start_time = time.time()
# solver.stop_conditions = [statistics.rms]
# solver.stop_values = [0.2]
# solver.solve(mod, steps = steps)
# print("--- %s seconds ---" % (time.time() - start_time))
# mod.plot2d(index=(0,1), data_type='detector_geometry')
# #mod.plot1d(index=0, data_type='detector_geometry')
#
# # pipe = pipeline.Pipeline(mod)
# # r = rescale.Rescale((80, 80, 80))
# # pipe.add_transform(r)
# # pipe.forward()
# mod.plot2d(index=(0,1))
# # pipe.backward()
# # mod.plot2d(index=(0,1))
# # #mod.plot1d(index=1, data_type='detector_geometry')
# # mod.plot1d(index=1)
# print(mod)
# # print(len(solver.statistics))