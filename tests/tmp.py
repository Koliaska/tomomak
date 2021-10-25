import tomomak.util.geometry3d_trimesh
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
import time

#Anaconda3-5.3.1-Windows-x86_64.exe
#conda upgrade -n base -c defaults --override-channels conda


# conda remove vtk conda remove mayavi and installing with pip install vtk and pip install mayavi. Thanks for your help!




from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d, objects3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
from tomomak.util import geometry3d_basic
import os
import tomomak.constraints.basic
import trimesh
import tomomak.detectors.detectors3d as detectors3d
from scipy.spatial.transform import Rotation as R

from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian, polar, toroidal, level
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import numpy as np
import tomomak.util.eqdsk as eqdsk
from tomomak import util
#from mayavi import mlab

import inspect
# import pyvista as pv
from mayavi import mlab


g = eqdsk.read_eqdsk('gglobus32994.g', b_ccw=-1)
eqdsk.psi_to_rho(g)
# edges=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
axes = [toroidal.Axis1d(radius=g['raxis'], name="theta", units="rad", size=2, upper_limit=np.pi/2),
        level.Axis1d(level_map=g['rho'], x=g['r'], y=g['z'], x_axis=g['raxis'], y_axis=g['zaxis'], bry_level=0.999,
                     name="rho", units="a.u.",edges=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.999]),
        polar.Axis1d(name="theta", units="rad", size=20)]
# axes = [polar.Axis1d(name="phi", units="rad", size=12),
#         cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=10)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
real_solution = objects2d.ellipse(m, ax_len=(1, 1), index=(1,2),center=(0.36,0), broadcast=False)
res= tomomak.util.array_routines.broadcast_object(real_solution, (1,2), m.shape)
res = tomomak.util.array_routines.normalize_broadcasted(res, (1,2), m, 'solution')
vols = util.geometry3d_trimesh.cell_volumes(m)
#res = res / vols
#noise = np.random.normal(0, 0.0001, real_solution.shape)

#res = util.array_routines.multiply_along_axis(res,np.linspace(22,10, num=axes[1].size), axis=1)
mod.solution = res

mod.plot2d(style ='colormesh', cartesian_coordinates=True, index=(1,2))
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)

axes = [toroidal.Axis1d(radius=15, name="theta", units="rad", size=7, upper_limit=np.pi),
        polar.Axis1d(name="phi", units="rad", size=8),
        cartesian.Axis1d(name="R", units="cm", size=9, upper_limit=10)]

m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
real_solution = objects2d.ellipse(m, ax_len=(5.2, 5.2), index=(1,2)) #/ geometry3d.cell_volumes(m)
noise = np.random.normal(0, 0.05, real_solution.shape)

mod.solution = real_solution #+ noise
mod.plot2d(style ='colormesh', cartesian_coordinates=True, index=(1,2))
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)
# b = pv.Box()
# print(b.volume)
# print(b.faces)
# print(b.bounds)
# vertices = np.array([[0, 0, 0],
#                      [1, 0, 0],
#                      [1, 1, 0],
#                      [0, 1, 0],
#                      [0.5, 0.5, -1]])
#
# # mesh faces
# faces = np.hstack([[4, 0, 1, 2, 3],  # square
#                    [3, 0, 1, 4],     # triangle
#                    [3, 1, 2, 4]])    # triangle
#
# surf = pv.PolyData(vertices, faces)
#
# # plot each face with a different color
# surf.plot(scalars=np.arange(3), cpos=[-1, 1, 0.5])



res = axes[0].cell_edges2d_cartesian(axes[1])
res = np.array(res[1][1])
# plt.plot(res[:,0], res[:, 1], 'g^')
# plt.show()


res = objects2d.ellipse(m, center=(0.36,0.0), ax_len=(0.2, 0.2), index=(0, 1), broadcast=False)
real_solution = tomomak.util.geometry3d_trimesh.broadcast_2d_to_3d(res, m, (0, 1), 2, 'solution')
mod.solution = real_solution
mod.plot2d()
# mod.plot3d(cartesian_coordinates=True, axes=True, style = 1)
# mod.plot3d(cartesian_coordinates=True, axes=True, style = 2)
# mod.plot3d(cartesian_coordinates=True, axes=True, style = 3)
# real_solution = tomomak.util.array_routines.broadcast_object(res, (0,), m.shape)

# real_solution  = tomomak.util.array_routines.normalize_broadcasted(real_solution, (0,),  m, 'solution')
# real_solution = objects2d.ellipse(m, ax_len=(5.2, 5.2), index=(0,1)) #/ geometry3d.cell_volumes(m)
# noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution #+ noise
det_geom = geometry2d.broadcast_1d_to_2d(res, m, 1, 0, 'detector_geometry')
mod.detector_geometry = [det_geom]
mod.plot2d(cartesian_coordinates=True, data_type='detector_geometry_n')
mod.plot2d(cartesian_coordinates=True, axes=True)
mod.plot1d(index=0)
mod.plot1d(index=1)
mod.plot2d(cartesian_coordinates=True, axes=True)

axes[0].RESOLUTION3D = 3

axes = [toroidal.Axis1d(radius=15, name="theta", units="rad", size=15, upper_limit=np.pi),
        polar.Axis1d(name="phi", units="rad", size=12),
        cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=10)]
axes[0].RESOLUTION3D = 3
axes[1].RESOLUTION2D = 3
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
real_solution = objects2d.ellipse(m, ax_len=(5.2, 5.2), index=(1,2)) #/ geometry3d.cell_volumes(m)
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution + noise
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)

axes = [polar.Axis1d(name="phi", units="rad", size=15),
        cartesian.Axis1d(name="R", units="cm", size=20, upper_limit=10),
        cartesian.Axis1d(name="Z", units="cm", size=21, upper_limit=40)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)

# Now let's see what the circle will look like in this coordinate system
real_solution = objects2d.ellipse(m, ax_len=(5.2, 5.2))
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution# + noise/10

mod.plot3d(style=0, cartesian_coordinates=True)


axes = [cartesian.Axis1d(name="X", units="cm", size=21, upper_limit=10),
        cartesian.Axis1d(name="Y", units="cm", size=22, upper_limit=10),
        cartesian.Axis1d(name="Z", units="cm", size=23, upper_limit=10)]
mesh = Mesh(axes)
mod = Model(mesh=mesh)

# The next step - is to create a synthetic object.
# Let's create 2D circle. Since our mesh is 3D it will be automatically broadcasted to 3rd dimension,
# so we get a cylinder.
real_solution = objects2d.ellipse(mesh, (5, 5), (3, 3))
# Now let's add some noise to this object in order to simulate realistic distribution.
noise = np.random.normal(0, 0.05, real_solution.shape)
mod.solution = real_solution + noise/2
# Now it's time to do plotting
mod.detector_geometry = np.array([mod.solution, mod.solution + noise/5, mod.solution + noise / 2])
mod.plot3d(style=1)
mod.plot3d(data_type='detector_geometry', equal_norm=True, style=0, axes=True)


# k = 50
# x = []
# y = []
# z = []
# faces = None
# f = np.array([[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
# for i in range(k):
#     x.extend([i*5, i*5, i*5, i*5  + 10, ])
#     y.extend([i*5, i*5, i*5+10, i*5, ])
#     z.extend([i*5, i*5+10, i*5, i*5, ])
#     if faces  is None:
#         faces = np.array([[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
#     else:
#         faces2 = f + i*4
#         faces = np.append(faces, faces2, axis=0)
#
#
#
# print(x, faces)
# mlab.triangular_mesh(x, y, z, faces, scalars=x)
# mlab.show()
# let's start by creating a 3D cartesian coordinate system.
# Since 3D tomography is much slower than 2D, in the example we will use 10x10x10 grid,
# but you can increase these values.
axes = [cartesian.Axis1d(name="X", units="cm", size=12, upper_limit=12),
        cartesian.Axis1d(name="Y", units="cm", size=11, upper_limit=11),
        cartesian.Axis1d(name="Z", units="cm", size=10, upper_limit=10)]
mesh = mesh.Mesh(axes)
mod = model.Model(mesh=mesh)
mod.detector_geometry = detectors3d.four_pi_detector_array(mesh, focus_point=(0,0, 0),
                                                           radius=45, theta_num=2, phi_num=3)


# geometry3d.show_cell(mesh, cell_index=(1, 1, 1))
# Now let's create a test object. It will be a sphere-like source with 1/R^2 density.
# Note that several object types are supported, including arbitrary objects, defined by vertices and faces,
# however calculation of the most objects intersection with each grid cell will take significant time.
# This is the price to pay for the versatility of the framework.
real_solution = objects2d.ellipse(mesh, (0, 0), (4, 9), index=(0, 1))
mod.solution = real_solution
mod.plot2d(index=(0,1))
real_solution = objects2d.ellipse(mesh, (0, 0), (4, 9), index=(1, 0))
real_solution = objects2d.ellipse(mesh, (0, 0), (4, 9), index=(2, 1))
mod.solution = real_solution
mod.plot2d(index=(2,1))
mod.plot2d(index=(0,1))
# Let's plot the data in different ways.
mod.solution = real_solution
mod.plot2d(index=(2,1))
mod.plot3d(axes=True, style=3, cartesian_coordinates=True)
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
mod.detector_geometry = detectors3d.four_pi_detector_array(mesh, focus_point=(0,0, 0),
                                                           radius=45, theta_num=20, phi_num=20)

mod.plot3d(axes=True, style=3, data_type='detector_geometry', cartesian_coordinates=True)
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
mod.save("tor.tmm")
# Now we can see the solution.
mod.plot3d(axes=True, style=3, cartesian_coordinates=True)
mod.plot2d(index=(1,2), cartesian_coordinates=True)

solver.plot_statistics()
# Of course, the result we got is not ideal due to limitations of our synthetic experiment.
# Luckily there are ways to improve our calculations, described in the more advanced examples.


axes = [toroidal.Axis1d(radius = 10, name="theta", units="rad", size=7),
        polar.Axis1d(name="phi", units="rad", size=12),
        cartesian.Axis1d(name="R", units="cm", size=8, upper_limit=9)]

mesh = mesh.Mesh(axes)
vertices, faces = axes[0].cell_edges3d_cartesian(axes[1], axes[2])
tomomak.util.geometry3d_trimesh.show_cell(mesh, cell_index=(3, 3, 3))

#
# And that's it. You get solution, which is, of course, not perfect,
# but world is not easy when you work with the limited data.
# There are number of ways to improve your solution, which will be described in other examples.




# if __name__ == "__main__":
#
#     import scipy.spatial.transform.rotation as rotation
#     ver = np.array([[0, 0, 0], [0, 0, 10], [0, 10, 0], [10, 0, 0]])
#     faces = np.array([[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
#     a= [[0,0, -1]]
#     b = [[0,-1,0]]
#
#
#     axes = [cartesian.Axis1d(name="X", units="cm", size=5, upper_limit=10),
#             cartesian.Axis1d(name="Y", units="cm", size=5, upper_limit=10),
#             cartesian.Axis1d(name="Z", units="cm", size=5, upper_limit=10)]
#     mesh1 = trimesh.Trimesh(vertices=ver, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
#
#
#     mesh = mesh.Mesh(axes)
#     mod = model.Model(mesh=mesh)
#     start = time.time()
#     mod.detector_geometry = detectors3d.fan_detector(mesh, (-5, 5, 5), [[5, 2, 6], [5, 7, 6], [5, 2, 4], [5, 7, 4]], number=[4,4], divergence=0.3)
#     end = time.time()
#     print("Time serial: ", end - start)
#     os.environ["TM_MP"] = "4"
#     start = time.time()
#     mod.detector_geometry = detectors3d.fan_detector(mesh, (-5, 5, 5), [[5, 2, 6], [5, 7, 6], [5, 2, 4], [5, 7, 4]], number=[4,4], divergence=0.3)
#     end = time.time()
#     print("Time parallel: ", end - start)
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = detectors3d.fan_detector(mesh, (-5, 5, 5), [[5, 2, 5], [5, 7, 5]], number=3, divergence=0.3)
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = [detectors3d.aperture_detector(mesh, [(4,-5,4), (6,-5,4), (4,-5,6), (6, -5, 6)], [(4,0,4), (6,0,4), (4,0,6), (6, 0, 6)], radius_dependence=False)]
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = [detectors3d.custom_detector(mesh, ver,radius_dependence=False) ]
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = [detectors3d.line_detector(mesh, (5,5,12), (10,2,0), 1, calc_volume=True,radius_dependence=False)]
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = [detectors3d.cone_detector(mesh, (5, 5, 20), (5,5,0), 0.3,  radius_dependence=True)]
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = [detectors3d.line_detector(mesh, (5,5,12), (10,2,0), None, calc_volume=False,radius_dependence=False)]
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.detector_geometry = detectors3d.four_pi_detector_array(mesh, (5, 5, 5), 20, 20, 10)
#     mod.plot3d(axes=True, style=3, data_type='detector_geometry')
#     mod.solution = np.log(objects3d.point_source(mesh, (5.5, 5.5,5.5))*100)
#     mod.plot3d(axes=True, style=3)
#     mod.solution = objects3d.trimesh_create(mesh, 'box',  extents=(6,16,12))
#     print(mod.solution)
#     mod.plot3d(axes=True, style=3)
#
#
#     trm = geometry3d.get_trimesh_grid(mesh)
#     mod.solution = geometry3d.grid_ray_intersection(trm, (-1.2, -1.2, -1.2), (10,10,10))
#     mod.plot3d(axes=True, style=3)
#     print(mod.solution)
#
#     mod.solution = geometry3d.grid_intersection3d(trm, ver, faces)
#     mod.plot3d(axes=True, style=3)
#     print(mod.solution)
#
#     ver = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [1, 0, 0]])
#     mesh1 = trimesh.Trimesh(vertices=ver, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
#     mesh1.show()
#     ver2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0.5, 0], [3, 0, 0]])
#     mesh2 = trimesh.Trimesh(vertices = ver + 0.2, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
#     # mesh2.show()
#     inter = trimesh.boolean.union((mesh1, mesh2))
#     # inter.show()
#     print (inter.volume)
#     trimesh.Scene([mesh1, mesh2]).show()
    # mesh1 = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=ver))
    # mesh1.show()
    # ver2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0.5, 0], [3, 0, 0]])
    # mesh2 = trimesh.convex.convex_hull( trimesh.Trimesh(vertices = ver + 0.2))
    # mesh2.show()
    # inter = trimesh.boolean.union((mesh1, mesh2))
    # inter.show()
    # print (inter.volume)

    # axes = [cartesian.Axis1d(name="X", units="cm", size=50, upper_limit=10),
    #         cartesian.Axis1d(name="Y", units="cm", size=50, upper_limit=10)]
    # mesh = mesh.Mesh(axes)
    # mod = model.Model(mesh=mesh)
    # from tomomak.detectors import detector_array
    #
    # os.environ["TM_MP"] = "4"
    # kw_list = []
    # for i in range(100):
    #     kw_list.append({'mesh': mesh, 'p1': (-5, 0), 'p2': (15, 15), 'width': 0.5, 'divergence': 0.1})
    #     kw_list.append({'mesh': mesh, 'p1': (-5, 5), 'p2': (15, 5), 'width': 0.5, 'divergence': 0.1})
    # det = detector_array.detector_array(func_name='tomomak.detectors.detectors2d.detector2d', kwargs_list=kw_list)
    # mod.detector_geometry = det
    # # mod.plot2d(data_type='detector_geometry')
    # os.environ["TM_MP"] = "4"
    # det = detector_array.detector_array(func_name='tomomak.detectors.detectors2d.detector2d', kwargs_list=kw_list)
    # mod.detector_geometry = det
    #
    # #real_solution = objects2d.ellipse(mesh, (5,5),(3,3))
    # import time
    #
    # start = time.time()
    #
    # det = detectors2d.fan_detector_array(mesh=mesh,
    #                                      focus_point=(5, 5),
    #                                      radius=11,
    #                                      fan_num=8,
    #                                      line_num=40,
    #                                      width=1,
    #                                      divergence=0.2)
    # end = time.time()
    # print("serial: ", end - start)
    # mod.detector_geometry = det
    # mod.plot2d(data_type='detector_geometry')
    # start = time.time()
    # os.environ["TM_MP"] = "4"
    # det = detectors2d.fan_detector_array(mesh=mesh,
    #                                         focus_point=(5, 5),
    #                                         radius=11,
    #                                         fan_num=8,
    #                                         line_num=10,
    #                                         width=1,
    #                                         divergence=0.2)
    # end = time.time()
    # print("parallel: ", end - start)
    # mod.detector_geometry = det
    # mod.plot2d(data_type='detector_geometry')
# det = detectors2d.two_pi_detector_array(mesh, (5,5), 10, 40)
# print(det)
# # Now we can calculate signal of each detector.
# # Of course in the real experiment you measure detector signals so you don't need this function.
# det_signal = signal.get_signal(real_solution, det)
# mod.detector_signal = det_signal
# mod.detector_geometry = det
# mod.plot2d(data_type='detector_geometry', equal_norm=True)
# solver = Solver()
#
# os.environ["TM_GPU"] = "1"
# solver.statistics = [statistics.RMS()]
# solver.constraints = [tomomak.constraints.basic.Positive()]
#
# solver.real_solution = real_solution
#
# solver.iterator = algebraic.SIRT()
# solver.iterator.alpha = np.full(100000, 0.1)
# steps = 1
# solver.solve(mod, steps=steps)
# solver.plot_statistics()
#
# os.environ["TM_GPU"] = "0"
# mod.solution = None
# solver.iterator = algebraic.SIRT()
# solver.constraints = [tomomak.constraints.basic.Positive()]
# solver.statistics = [statistics.RMS()]
# solver.iterator.alpha = np.full(100, 0.1)
# solver.solve(mod, steps=steps)
# solver.plot_statistics()
#
# mod.plot2d(data_type='detector_geometry')
# # solver.plot_statistics()
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