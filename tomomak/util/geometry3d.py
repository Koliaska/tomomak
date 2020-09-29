import numpy as np
from scipy.interpolate import LinearNDInterpolator
import itertools
import trimesh
import os
import subprocess
import sys
from collections.abc import Iterable
from .geometry2d import check_spatial
from . import geometry2d
from . import array_routines
from tomomak.mesh import cartesian


def get_trimesh_grid(mesh, index=(0, 1, 2)):
    """Get array of trimesh objects (3D meshes) corresponding to each grid cell.

    Helper function. Trimesh objects are used for complex 3d calculations.
    Only axes which implements cell_edges3d_cartesian method are supported (see abstract_axes.py).
    Trimesh module is used for the calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.:
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)

    Returns:
        list: 1D/2D or 3D list (depending on used tomomak mesh), containing trimesh representation of each cell.

    Raises:
        TypeError if no combination of axes supports the cell_edges3d_cartesian() method or all axes dimensions are > 3.
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    if mesh.axes[index[0]].dimension == 3:
        try:
            (vertices, faces) = mesh.axes[index[0]].cell_edges3d_cartesian()
            shape = mesh.axes[index[0]].size
            trimesh_list = np.zeros(shape).tolist()
            for i, _ in enumerate(trimesh_list):
                if faces is None:
                    cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i]))
                else:
                    cell = trimesh.Trimesh(vertices=vertices[i], faces=faces[i])
                trimesh_list[i] = cell
        except (TypeError, AttributeError) as e:
            raise type(e)(e.message + "Custom axis should implement cell_edges3d_cartesian method. "
                                      " See docstring for more information.")
    # If 1st axis is 2D
    elif mesh.axes[index[0]].dimension == 2 or mesh.axes[index[1]].dimension == 2:
        (vertices, faces) = mesh.axes_method2d(index, "cell_edges3d_cartesian")
        vertices = vertices.tolist()
        faces = faces.tolist()
        shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size)
        trimesh_list = np.zeros(shape).tolist()
        for i, row in enumerate(trimesh_list):
            for j, col in enumerate(row):
                if faces is None:
                    cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i][j]))
                else:
                    cell = trimesh.Trimesh(vertices=vertices[i][j], faces=faces[i][j])
                trimesh_list[i][j] = cell
    # If axes are 1D
    elif mesh.axes[index[0]].dimension == mesh.axes[index[1]].dimension == mesh.axes[index[2]].dimension == 1:
        (vertices, faces) = mesh.axes_method3d(index, "cell_edges3d_cartesian")
        vertices = vertices.tolist()
        faces = faces.tolist()
        shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size, mesh.axes[index[2]].size)
        trimesh_list = np.zeros(shape).tolist()
        for i, row in enumerate(trimesh_list):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    if faces is None:
                        cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i][j][k]))
                    else:
                        cell = trimesh.Trimesh(vertices=vertices[i][j][k], faces=faces[i][j][k])
                    trimesh_list[i][j][k] = cell
    else:
        raise TypeError("3D objects can be built on the 1D,2D and 3D axes only.")

    return trimesh_list


def get_trimesh_obj(vertices, faces=None):
    """Get trimesh object with given faces and vertices.

    Args:
        vertices (array-like): a list of lists of points (x, y, z) in cartesian coordinates.
        faces (array-like): a list of lists of cell faces. Each face is a list of vertices, denoted in cw direction.
            If faces is None, faces are created automatically for the convex hull of the vertices. Default:  None.

    Returns:
        Trimesh: trimesh representation of the object.
    """
    if faces is None:
        obj3d = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices))
    else:
        obj3d = trimesh.Trimesh(vertices=vertices, faces=faces)
    return obj3d


def grid_intersection3d(trimesh_list, obj3d):
    """Create array, representing intersection volume of each object in list  with given object.

    List of object is usually a list, containing each cell of the used 3D grid.
    The object is usually a real-world object or detector line of sight.
    Very slow.

    Args:
        trimesh_list: a list  containing trimesh representation of each cell (e.g. obtained with get_trimesh function).
        obj3d(Trimesh): trimesh representation of the object.

    Returns:
        ndarray: numpy array, representing intersection volume of each object in list  with given object.

    Raises:
        TypeError if trimesh_list has 0 or > 3 dimensions.
    """
    list_dim = len(_dim(trimesh_list))
    if list_dim == 3:
        shape = (len(trimesh_list), len(trimesh_list[0]), len(trimesh_list[0][0]))
        res = np.zeros(shape)
        sys.stderr = open(os.devnull, "w")  # supressing console for trimesh calculations
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    try:
                        inters = trimesh.boolean.intersection((obj3d, trimesh_list[i][j][k]))
                        inters = inters.volume
                    except (subprocess.CalledProcessError, ValueError):
                        inters = 0
                    res[i, j, k] = inters
        sys.stderr = sys.__stderr__
        return res
    elif list_dim == 2:
        shape = (len(trimesh_list), len(trimesh_list[0]))
        res = np.zeros(shape)
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                try:
                    inters = trimesh.boolean.intersection((obj3d, trimesh_list[i][j])).volume
                except subprocess.CalledProcessError:
                    inters = 0
                res[i, j] = inters
        return res
    elif list_dim == 1:
        shape = (len(trimesh_list))
        res = np.zeros(shape)
        for i, row in enumerate(res):
            try:
                inters = trimesh.boolean.intersection((obj3d, trimesh_list[i])).volume
            except subprocess.CalledProcessError:
                inters = 0
            res[i] = inters
        return res
    else:
        raise TypeError("trimesh_list should be a 1D or 2D or 3D list.")


def _dim(lst):
    if not type(lst) == list:
        return []
    return [len(lst)] + _dim(lst[0])


def grid_ray_intersection(trimesh_list, p1, p2):
    """Create array, representing intersection of each object in list  with given ray.

    List of object is usually a list, containing each cell of the used 3D grid.
    The ray is usually a  detector line of sight.
    If ray intersects the cell, than value in the returned array = cell volume, otherwise it is 0.

    Args:
        trimesh_list: a list  containing trimesh representation of each cell (e.g. obtained with get_trimesh function).
        p1 (tuple of 3 floats): Ray origin (x, y, z).
        p2 (tuple of 3 floats): Second point, characterizing ray direction (x, y, z).

    Returns:
          ndarray: numpy array, representing intersection of each object in list  with given ray.
    """
    list_dim = len(_dim(trimesh_list))
    if list_dim == 3:
        shape = (len(trimesh_list), len(trimesh_list[0]), len(trimesh_list[0][0]))
        res = np.zeros(shape)
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    res[i, j, k] = _cell_ray_inters(trimesh_list[i][j][k], p1, p2)
        return res
    elif list_dim == 2:
        shape = (len(trimesh_list), len(trimesh_list[0]))
        res = np.zeros(shape)
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                res[i, j] = _cell_ray_inters(trimesh_list[i][j], p1, p2)
        return res
    elif list_dim == 1:
        shape = (len(trimesh_list), len(trimesh_list[0]))
        res = np.zeros(shape)
        for i, row in enumerate(res):
            res[i] = _cell_ray_inters(trimesh_list[i], p1, p2)
        return res
    else:
        raise TypeError("trimesh_list should be a 1D or 2D or 3D list.")


def _cell_ray_inters(trimesh_element, p1, p2):
    try:
        direction = (np.array(p2)-np.array(p1),)
        inters = int(trimesh_element.ray.intersects_any((p1,), direction))
    except subprocess.CalledProcessError:
        inters = 0
    return inters


def cell_volumes(mesh, index=(0, 1, 2)):
    """Get array of each cell volumes.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        index(tuple of one, two or three ints, optional): axes to build object at. Default: (0, 1, 2)

    Returns:
        ndarray: numpy array, representing volume of each cell.
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    trimesh_list = get_trimesh_grid(mesh, index)
    volumes = np.zeros_like(trimesh_list)
    for idx, value in np.ndenumerate(trimesh_list):
        volumes[idx] = value.volume
    return np.array(volumes, dtype=float)


def cell_distances(mesh, p, index=(0, 1, 2)):
    """Get distance to each cell on 3D mesh.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p (list of three floats): list representing a point in 3D coordinates.
        index (tuple of 1/2/3 ints, optional): axes to calculate distance at. Default:  (0,1,2)

    Returns:
        ndarray: array with distances.

    Raises
        TypeError if axis dimension is > 3.
        """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    p = np.array(p)
    if mesh.axes[index[0]].dimension == 3:
        try:
            cell_coordinates = mesh.axes[index[0]].cartesian_coordinates()
            shape = mesh.axes[index[0]].size
            distances = np.zeros(shape)
            for i, _ in enumerate(distances):
                distances[i] = np.sqrt(np.sum(
                    (np.array([c - cell_coordinates[ind][i] for ind, c in enumerate(p)]) ** 2)))
        except (TypeError, AttributeError) as e:
            raise type(e)(e.message + "Custom axis should implement cartesian_coordinates method. "
                                      " See docstring for more information.")
    # If one axis is 2D
    elif mesh.axes[index[0]].dimension == 2 or mesh.axes[index[1]].dimension == 2:
        cell_coordinates = mesh.axes_method2d(index, "cartesian_coordinates")
        shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size)
        distances = np.zeros(shape)
        for i, row in enumerate(distances):
            for j, col in enumerate(row):
                distances[i, j] = np.sqrt(np.sum(
                    (np.array([c - cell_coordinates[ind][i, j] for ind, c in enumerate(p)]) ** 2)))
    # If axes are 1D
    elif mesh.axes[index[0]].dimension == mesh.axes[index[1]].dimension == mesh.axes[index[2]].dimension == 1:
        cell_coordinates = mesh.axes_method3d(index, "cartesian_coordinates")
        shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size, mesh.axes[index[2]].size)
        distances = np.zeros(shape)
        for i, row in enumerate(distances):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    distances[i, j, k] = np.sqrt(np.sum(
                        (np.array([c - cell_coordinates[ind][i, j, k] for ind, c in enumerate(p)]) ** 2)))
    else:
        raise TypeError("3D objects can be built on the 1D,2D and 3D axes only.")
    return distances


def make_regular(data, x_grid, y_grid, z_grid, interp_size):
    """Convert irregular 3D grid to regular.

        The x, y and z arrays are then supposed to be like arrays generated by numpy.meshgrid,
        but not necessarily regularly spaced.
    Args:
        data (3D ndarray): Data to interpolate.
        x_grid (3D ndarray): X grid.
        y_grid (3D ndarray): Y grid.
        z_grid (3D ndarray): Z grid.
        interp_size (int): The new grid wil have  interp_size * interp_size * interp_size dimensions.

    Returns:
        4 3D ndarrays: new x,y,z grids and new data
    """
    points = np.transpose(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
    new_data = data.flatten()
    interp = LinearNDInterpolator(points, new_data, fill_value=0)
    coords = []
    for c in (x_grid, y_grid, z_grid):
        c_s = np.sort(c.flatten())
        c_new = np.linspace(c_s[0] - (c_s[1] - c_s[0]), c_s[-1] + (c_s[-1] - c_s[-2]), interp_size)
        coords.append(c_new)
    x_grid, y_grid, z_grid = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    new_points = np.transpose(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
    new_data = interp(new_points)
    new_data = np.reshape(new_data, x_grid.shape)
    return x_grid, y_grid, z_grid, new_data


def show_cell(mesh, index=(0, 1, 2), cell_index=(0, 0, 0)):
    """Shows 3D borders of the given cell.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        index(tuple of one, two or three ints, optional): axes to build object at. Default: (0, 1, 2)
        cell_index(tuple of one, two or three ints, optional): index of the cell. Default: (0, 0, 0)

    Returns:
        None
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    trimesh_list = get_trimesh_grid(mesh, index)
    for i in cell_index:
        trimesh_list = trimesh_list[i]
    trimesh_list.show()


def trimesh_transform_matrix(p1, p2, shift):
    """Create trimesh 4x4 transformation matrix suitable for trimesh creation routines.

    Args:
        p1 (tuple of 3 floats): Detector origin (x, y, z).:
        p2 (tuple of 3 floats): Detector line of sight direction(x, y, z).
        shift (float): shift  in z direction of the trimesh object center.

    Returns:
        4x4 ndarray: transformation matrix.
    """
    v1 = np.array((0, 0, -1))
    v2 = np.array(p2) - np.array(p1)
    rot_matr = rotation_matrix_from_vectors(v1, v2)
    rot_vector = np.dot(rot_matr, v1) * shift
    rot_vector += np.array(p1)
    shift = np.array([rot_vector, ]).transpose()
    footer = np.array([[0, 0, 0, 1]])
    transform_matrix = np.append(rot_matr, shift, axis=1)
    transform_matrix = np.append(transform_matrix, footer, axis=0)
    return transform_matrix


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2

    From stackoverflow
    Args:
        vec1: A 3d "source" vector
        vec2: A3d "destination" vector

    Returns:
         A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def mesh_center(mesh, index=(0, 1, 2)):
    """Find geometrical center of the given mesh in the cartesian coordinates.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        index(tuple of one, two or three ints, optional): axes to build object at. Default: (0, 1, 2)

    Returns:
        list: (x, y, z) coordinates of the mesh center.
    """
    check_spatial(mesh, index)
    vertices, _ = mesh.axes_method3d(index, 'cell_edges3d_cartesian')
    min_v = np.array(vertices[0, 0, 0][0])
    max_v = np.array(vertices[0, 0, 0][0])
    for i in vertices:
        for j in i:
            for k in j:
                for v in k:
                    min_v = np.maximum(max_v, v)
                    max_v = np.minimum(min_v, v)
    return (min_v + max_v) / 2


def _convert_slice_cart(data, mesh, index, direct):
    if isinstance(index, int):
        index = [index]
    if len(index) != len(data.shape):
        raise ValueError('index and data have different sizes')
    axes = []
    for i in index:
        axes.append((mesh.axes[i]))
    if any([type(a) is not cartesian.Axis1d for a in axes]):
        if mesh.axes[index[0]].dimension in (3, 2, 1):
            ax_num = 4 - mesh.axes[index[0]].dimension
        else:
            raise ValueError('Incorrect axes dimensions')
        volumes = np.array(cell_volumes(mesh, index), dtype=float)
        for i in range(ax_num):
            dv = 1 / mesh.axes[index[i]].volumes
            volumes = array_routines.multiply_along_axis(volumes, dv, i)
        if direct:
            new_data = data / volumes
        else:
            new_data = data * volumes
    else:
        return data
    return new_data


def convert_slice_from_cartesian(data, mesh, index, data_type):
    """Converts 3D slice in cartesian coordinates to mesh coordinates.

    Args:
        data (numpy array): 3D slice of solution or detector geometry.
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints): axes corresponding to data.
        data_type(str): 'solution' or 'detector_geometry'.

    Returns:
        numpy array: converted data.
    """
    check_spatial(mesh, index)
    if data_type == 'solution':
        return _convert_slice_cart(data, mesh, index, False)
    elif data_type == 'detector_geometry':
        return _convert_slice_cart(data, mesh, index, True)
    else:
        raise ValueError("Wrong data_type.")


def convert_slice_to_cartesian(data, mesh, index, data_type):
    """Converts 3D slice in  mesh coordinates to cartesian coordinates.

    Args:
        data (numpy array): 3D slice of solution or detector geometry.
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints): axes corresponding to the data.
        data_type(str): 'solution' or 'detector_geometry'.

    Returns:
        numpy array: converted data.
    """
    if data_type == 'solution':
        return _convert_slice_cart(data, mesh, index, True)
    elif data_type == 'detector_geometry':
        return _convert_slice_cart(data, mesh, index, False)
    else:
        raise ValueError("Wrong data_type.")


def _convert_cart(data, mesh, index, direct):
    if isinstance(index, int):
        index = [index]
    if len(data.shape) != len(mesh.axes):
        raise ValueError('data and mesh have different sizes')
    axes = []
    for i in index:
        axes.append((mesh.axes[i]))
    if any([type(a) is not cartesian.Axis1d for a in axes]):
        if mesh.axes[index[0]].dimension in (3, 2, 1):
            ax_num = 4 - mesh.axes[index[0]].dimension
        else:
            raise ValueError('Incorrect axes dimensions')
        c_areas = cell_volumes(mesh, index)
        for i in range(ax_num):
            dv = 1 / mesh.axes[index[i]].volumes
            c_areas = array_routines.multiply_along_axis(c_areas, dv, i)
        last_ind = list(range(-ax_num, 0))
        new_data = np.moveaxis(data, index, last_ind)
        if direct:
            new_data = new_data / c_areas
        else:
            new_data = new_data * c_areas
        new_data = np.moveaxis(new_data, last_ind , index)
    else:
        return data
    return new_data


def convert_from_cartesian(data, mesh, index, data_type):
    """Converts data in cartesian coordinates to mesh coordinates.

    Args:
        data (numpy array): solution or detector geometry or their slice .
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints): axes corresponding to the data.
        data_type(str): 'solution' or 'detector_geometry'.
    Returns:
        numpy array: converted data.
    """
    check_spatial(mesh, index)
    if data_type == 'solution':
        return _convert_cart(data, mesh, index, False)
    elif data_type == 'detector_geometry':
        return _convert_cart(data, mesh, index, True)
    else:
        raise ValueError("Wrong data_type.")


def convert_to_cartesian(data, mesh, index, data_type):
    """Converts data in mesh coordinates to cartesian coordinates.

    Args:
        data (numpy array): solution or detector geometry or their slice .
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints): axes corresponding to the data.
        data_type(str): 'solution' or 'detector_geometry'.

    Returns:
        numpy array: converted data.
    """
    if data_type == 'solution':
        return _convert_cart(data, mesh, index, True)
    elif data_type == 'detector_geometry':
        return _convert_cart(data, mesh, index, False)
    else:
        raise ValueError("Wrong data_type.")


def broadcast_2d_to_3d(data, mesh, index2d, index3, data_type):
    """Broadcasts 2d array to 3d array, taking into account mesh geometry.

    Function changes density (for solution) or volume (for detector geometry) in such a way,
    that proportionality along the firs axes is conserved in the cartesian coordinates.
    Plot solution or detector_geometry_n with cartesian_coordinates=True to check the result.
    Args:
        data (2d or 1d ndarray): data along the first axes.
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index2d (tuple): list of of the axes/axis, corresponding to the data array on the mesh.
        index3 (int): index of the broadcasted axis.
        data_type (str): 'solution' or 'detector_geometry'.

    Returns:
        ndarray: broadcasted data

    Examples:
        from tomomak.mesh import cartesian, polar, toroidal
        from tomomak.mesh import mesh
        from tomomak import model
        import numpy as np
        from tomomak.util import geometry3d
        from tomomak.test_objects import objects2d
        from tomomak.detectors import detectors2d

        axes = [toroidal.Axis1d(radius=20, name="theta", units="rad", size=15, upper_limit=np.pi),
                polar.Axis1d(name="phi", units="rad", size=12),
                cartesian.Axis1d(name="R", units="cm", size=11, upper_limit=10)]
        m = mesh.Mesh(axes)
        mod = model.Model(mesh=m)
        res = objects2d.ellipse(m, ax_len=(5.2, 5.2), index=(1, 2), broadcast=False)
        real_solution = geometry3d.broadcast_2d_to_3d(res, m, (1, 2), 0, 'solution')
        mod.solution = real_solution
        det = detectors2d.detector2d(m, (-50, -50), (50, 50), 40, index=(1, 2), radius_dependence=False, broadcast=False)
        det_geom = geometry3d.broadcast_2d_to_3d(det, m, (1,2), 0, 'detector_geometry')
        mod.detector_geometry = [det_geom]
        mod.plot3d(cartesian_coordinates=True, data_type='detector_geometry_n', limits=(0, 2))
        mod.plot3d(cartesian_coordinates=True)
    """
    if len(data.shape) == len(index2d) + 1:
        raise ValueError("Data is already broadcasted.")
    if index3 in index2d:
        raise ValueError('Index 3 cannot be equal to index 2 or index 1.')
    index3d = list(index2d)
    index3d.append(index3)
    index3d.sort()
    volumes = cell_volumes(mesh, index3d)
    areas = geometry2d.cell_areas(mesh, index2d)
    shape = (mesh.axes[0].size, mesh.axes[1].size, mesh.axes[2].size)
    if data_type == 'solution':
        new_data = data / areas
        new_data = array_routines.broadcast_object(new_data, index2d, shape)
        new_data = new_data * volumes
        new_data = array_routines.multiply_along_axis(new_data, 1 / mesh.axes[index3].volumes, index3)
    elif data_type == 'detector_geometry':
        new_data = array_routines.broadcast_object(data, index2d, shape)
        new_data = array_routines.multiply_along_axis(new_data, mesh.axes[index3].volumes, index3)
    else:
        raise ValueError('Unknown data type.')
    return new_data
