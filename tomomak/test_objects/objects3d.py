"""Functions for creation of different 3d objects.

Synthetic object are usually used to test different tomomak components.
"""
import numpy as np
import tomomak.util.array_routines
import tomomak.util.geometry3d_basic as geometry3d
import importlib

import tomomak.util.geometry3d_trimesh


def figure(mesh, vertices, faces=None, index=(0, 1, 2), density=1, broadcast=True):
    """Create a shape with arbitrary coordinates. Very slow.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        vertices: a list of lists of points (x, y, z) in cartesian coordinates.
        faces: a list of lists of cell faces. Each face is a list of vertices, denoted in cw direction.
            If faces is None, faces are created automatically for the convex hull of the vertices. Default:  None.
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)
        density (float, optional): Object density. E.g. number of emitted particles per second in 4*pi. Default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray:  numpy array, representing figure on the given mesh.
    """
    obj3d = tomomak.util.geometry3d_trimesh.get_obj(vertices, faces)
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
    trimesh_list = tomomak.util.geometry3d_trimesh.get_grid(mesh, index)
    intersection = tomomak.util.geometry3d_trimesh.grid_intersection3d(trimesh_list, obj3d)
    intersection *= density
    dv = tomomak.util.geometry3d_trimesh.cell_volumes(mesh, index)
    intersection /= dv
    intersection = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(intersection, mesh, index,
                                                                                data_type='solution')
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
    dv = tomomak.util.geometry3d_trimesh.cell_volumes(mesh, index)
    distances /= dv
    distances = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(distances, mesh, index,
                                                                             data_type='solution')
    if broadcast:
        distances = tomomak.util.array_routines.broadcast_object(distances, index, mesh.shape)
    return distances


def rotational_mode(mesh, n, m, coord, index, n_phase=0, m_phase=0, shift=0):
    """ Creates rotational mode in 3d coordinates.
    See tokomak (n,m) modes for the reference.
    Mode will have n maximums along 1st axis, m maximums along 2nd axis and exist only in specific cell along 3rd axis.
    Makes sense only when 1st and 2nd axes are polar or toroidal rotational axes.

    Args:

        mesh (tomomak.mesh.Mesh): mesh to work with.
        n (int): toroidal rotation number.
        m (int): polar rotation number
        coord (float): coordinate, at which mode exist.
        index (tuple of 3 ints): indexes of axes in the following order:
          (axis for n rotation, axis for m rotation, axis without rotation).
        n_phase (float): starting n rotation phase. Optional. Default: 0.
        m_phase (float): starting m rotation phase. Optional. Default: 0.
        shift (float): shift if the mode. Optional. Default: 0.

    Returns:
        result (3d ndarray): resulting 3d object.

    """

    # find cell index
    cell_edges = mesh.axes[index[2]].cell_edges
    if coord < min(cell_edges) or coord > max(cell_edges):
        raise ValueError("Coordinate lies outside of axis coordinates.")
    coord_index = 0
    for i, _ in enumerate(cell_edges):
        if cell_edges[i] <= coord < cell_edges[i + 1] \
                or cell_edges[i] > coord >= cell_edges[i + 1]:
            coord_index = i
            break

    # create structure

    n_structure = mesh.axes[index[0]].coordinates
    n_structure = n_structure * n + n_phase

    m_structure = mesh.axes[index[1]].coordinates * m + m_phase
    res_3d = np.zeros(mesh.shape)
    for ar_ind, x in np.ndenumerate(res_3d):
        coord_val = int(coord_index == ar_ind[index[2]])
        m_phase = m_structure[ar_ind[index[1]]]
        n_phase = n_structure[ar_ind[index[0]]]
        res_3d[ar_ind] = coord_val * np.cos(m_phase + n_phase) + coord_val * shift

    dv = tomomak.util.geometry3d_trimesh.cell_volumes(mesh)

    return res_3d * dv
