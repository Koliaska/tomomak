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
from tomomak.test_objects import objects2d, objects3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
from tomomak.util import geometry3d
import os
import tomomak.constraints.basic
import trimesh
import tomomak.detectors.detectors3d as detectors3d
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    x = np.array([[4, 2, 1],[1,0, 5]])
    print(np.argwhere(x == 0))
    import scipy.spatial.transform.rotation as rotation
    ver = np.array([[0, 0, 0], [0, 0, 10], [0, 10, 0], [10, 0, 0]])
    faces = np.array([[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
    a= [[0,0, -1]]
    b = [[0,-1,0]]


    axes = [cartesian.Axis1d(name="X", units="cm", size=10, upper_limit=10),
            cartesian.Axis1d(name="Y", units="cm", size=10, upper_limit=10),
            cartesian.Axis1d(name="Z", units="cm", size=10, upper_limit=10)]
    mesh1 = trimesh.Trimesh(vertices=ver, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])


    mesh = mesh.Mesh(axes)
    mod = model.Model(mesh=mesh)
    mod.detector_geometry = [detectors3d.line_detector(mesh, (5,5,12), (10,2,0), 0.2, calc_volume=True,radius_dependence=False)]
    mod.plot3d(axes=True, style=3, data_type='detector_geometry')
    mod.detector_geometry = detectors3d.four_pi_detector_array(mesh, (5, 5, 5), 20, 20, 10)
    mod.plot3d(axes=True, style=3, data_type='detector_geometry')
    mod.solution = np.log(objects3d.point_source(mesh, (5.5, 5.5,5.5))*100)
    mod.plot3d(axes=True, style=3)
    mod.solution = objects3d.trimesh_create(mesh, 'box',  extents=(6,16,12))
    print(mod.solution)
    mod.plot3d(axes=True, style=3)


    trm = geometry3d.get_trimesh_grid(mesh)
    mod.solution = geometry3d.grid_ray_intersection(trm, (-1.2, -1.2, -1.2), (10,10,10))
    mod.plot3d(axes=True, style=3)
    print(mod.solution)

    mod.solution = geometry3d.grid_intersection3d(trm, ver, faces)
    mod.plot3d(axes=True, style=3)
    print(mod.solution)

    ver = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [1, 0, 0]])
    mesh1 = trimesh.Trimesh(vertices=ver, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
    mesh1.show()
    ver2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0.5, 0], [3, 0, 0]])
    mesh2 = trimesh.Trimesh(vertices = ver + 0.2, faces=[[0, 1, 2], [3, 1, 0], [0, 2, 3], [2, 1, 3]])
    # mesh2.show()
    inter = trimesh.boolean.union((mesh1, mesh2))
    # inter.show()
    print (inter.volume)
    trimesh.Scene([mesh1, mesh2]).show()
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