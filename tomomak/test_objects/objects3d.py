"""Functions for creation of different 3d objects.

Synthetic object are usually used to test different tomomak components.
"""
import numpy as np
import tomomak.util.array_routines
import tomomak.util.geometry3d as geometry3d
import importlib


def figure(mesh, vertices, faces=None, index=(0, 1, 2), density=1, broadcast=True):
    """Create a shape with arbitrary coordinates. Very slow.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        vertices: a list of lists of points (x, y, z) in cartesian coordinates: vertices the cell.
        faces: a list of lists of cell faces. Each face is a list of vertices, denoted in cw direction.
            If faces is None, faces are created automatically for the convex hull of the vertices. Default:  None.
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)
        density (float, optional): Object density. E.g. number of emitted particles per second in 4*pi. Default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray:  numpy array, representing figure on the given mesh.
    """
    obj3d = geometry3d.get_trimesh_obj(faces, vertices)
    return _create_figure(mesh, obj3d, index, density, broadcast)


def trimesh_create(mesh, func_name, index=(0, 1, 2), density=1, broadcast=True, *args, **kwargs):
    """Create object using trimesh routines. Very slow.

    Different shapes - boxes, spheres, cones, etc are supported: see trimesh.creation.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        func_name(string): name of the trimesh creation routine
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)
        density (float, optional): Object density. E.g. number of emitted particles per second in 4*pi. Default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.
        *args: passed to Trimesh routine.
        **kwargs: passed to Trimesh routine.

    Returns:
        ndarray:  numpy array, representing figure on the given mesh.

     Examples:
         mod.solution = objects3d.trimesh_create(mesh, 'box',  extents=(6,16,12))
    """
    module = importlib.import_module('trimesh.creation')
    func = getattr(module, func_name)
    obj3d = func(*args, **kwargs)
    return _create_figure(mesh, obj3d, index, density, broadcast)


def _create_figure(mesh, obj3d, index, density, broadcast):
    trimesh_list = geometry3d.get_trimesh_grid(mesh, index)
    intersection = geometry3d.grid_intersection3d(trimesh_list, obj3d)
    intersection *= density
    if broadcast:
        intersection = tomomak.util.array_routines.broadcast_object(intersection, index, mesh.shape)
    return intersection


def point_source(mesh, point, index=(0, 1, 2), density=1, broadcast=True):
    """Create 3D array with densities ~ 1/R^2, where R - distance to the given point.

    R is defined as distance between the given point ond cell mesh center.
    If the given point coincides with with a cell center (and R = 0), R is set  = D^(2 ** 4/3),
    where D - distance to the closest cell.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        point (list of three floats): list representing point in 3D coordinates.
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)
        density (float, optional): Object density. E.g. number of emitted particles per second in 4*pi. Default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray:  numpy array, representing point source on the given mesh.

    """
    distances = geometry3d.cell_distances(mesh, point, index)
    zero = np.argwhere(distances == 0)
    if zero.size:
        sorted_dist = np.sort(distances, axis=None)
        distances[tuple(zero[0])] = sorted_dist[1] / (2 ** 4/3)  # distance to the radius,
        # limiting 1/2 volume of the sphere with R = 1/2 distance
    distances = 1 / (distances ** 2) * density
    if broadcast:
        distances = tomomak.util.array_routines.broadcast_object(distances, index, mesh.shape)
    return distances
