"""Functions for creation of different 2d objects.

Synthetic object are usually used to test different tomomak components.
"""
import numpy as np
import shapely.geometry
import shapely.affinity
import tomomak.util.array_routines
import tomomak.util.geometry2d
from tomomak.mesh import cartesian


def polygon(mesh, points=((0, 0), (5, 5), (10, 0)), index=(0, 1), density=1, broadcast=True):
    """Create solution array, representing 2d polygon, defined by specified points on the given mesh.

    If there are more than 2 dimension in model, broadcasting to other dimensions is performed.
    If broadcasting is not needed private method _polygon may be used.
    Only axes which implements cell_edges2d_cartesian method are supported.
    cell_edges2d_cartesian method should  accept second axe and return 2d list of ordered sequence of point tuples for two 1d axes
    or 1d list of ordered sequence of point tuples for one 2d axis.
    Each point tuple represents cell borders in the 2D cartesian coordinates.
    E.g. borders of the cell of two cartesian axes with edges (0,7) and (0,5)
    is a rectangle which can be represented by the following point tuple ((0 ,0), (0, 7), (5,7), (5, 0)).
    Shapely module is used for the calculation.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        points (An ordered sequence of point tuples, optional): Polygon points (x, y).
            default: ((0 ,0), (5, 5), (10, 0))
        index (tuple of two ints, optional): axes to build object at. Default: (0,1).
        density (float, optional): Object density. E.g. number of emitted particles per second in 4*pi. Default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray:  numpy array, representing polygon on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    if isinstance(index, int):
        index = [index]
    res = tomomak.util.geometry2d.intersection_2d(mesh, points, index)
    ds = tomomak.util.geometry2d.cell_areas(mesh, index)
    res /= ds
    res *= density
    res = tomomak.util.geometry2d.convert_slice_from_cartesian(res, mesh, index, data_type='solution')
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)
    return res


def rectangle(mesh, center=(0, 0), size=(10, 10), index=(0, 1), density=1, broadcast=True):
    """Create solution array, representing 2d rectangle, defined by specified parameters.

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        center (tuple of int, optional): Center of the rectangle, given by tuples with 2 elements(x, y). default: (0, 0).
        size (tuple of int, optional): Length and height of the rectangle,
            given by tuples with 2 elements(length, height). default: (10, 10).
        index (tuple of two ints, optional): axes to build object at. Default:  (0,1)
        density (float, optional): Object density. default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: 2D numpy array, representing rectangle on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    points = [(center[0] - size[0] / 2, center[1] - size[1] / 2), (center[0] + size[0] / 2, center[1] - size[1] / 2),
              (center[0] + size[0] / 2, center[1] + size[1] / 2), (center[0] - size[0] / 2, center[1] + size[1] / 2)]
    return polygon(mesh, points, index, density, broadcast)


def ellipse(mesh, center=(0, 0), ax_len=(5, 5), index=(0, 1), density=1, resolution=32, broadcast=True):
    """Create solution array, representing 2d real_solution, defined by specified parameters.

    Shapely module is used for calculation.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        center (tuple of int, optional): Center of the real_solution, given by tuples with 2 elements(x, y).
            Default: (0, 0).
        ax_len (tuple of int, optional): Half-width and Half-height of the real_solution,
            given by tuples with 2 elements (a, b). default: (5, 5).
        index (tuple of two ints, optional): axes to build object at. Default:  (0,1)
        density (float, optional): Object density. default: 1.
        resolution (integer, optional): Relative number of points, approximating real_solution. default: 32.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: 2D numpy array, representing real_solution on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    points = shapely.geometry.Point(0, 0).buffer(1, resolution)
    points = shapely.affinity.scale(points, ax_len[0], ax_len[1])
    points = shapely.affinity.translate(points, center[0], center[1])
    return polygon(mesh, points, index, density, broadcast)


def pyramid(mesh, center=(0, 0), size=(10, 10), index=(0, 1), height=1, broadcast=True):
    """Create solution array, representing  2d rectangle defined by specified parameters
        with density changing as height of the quadrangular pyramid .

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        center (tuple of int): Center of the pyramid, given by tuples with 2 elements(x, y). default: (0, 0).
        size (tuple of int): Length and height of the pyramid, given by tuples with 2 elements(length, height).
            default: (10, 10).
        index (tuple of two ints, optional): axes to build object at. Default:  (0,1)
        height (float, optional): Pyramid max height. Minimum height is 0. default: 1.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: 2D numpy array, representing pyramid on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    for i in index:
        if mesh.axes[i] is not cartesian.Axis1d:
            raise TypeError("such combination of axes is not supported.")
    rect = rectangle(mesh, center, size, index, height, broadcast=False)
    mask = np.zeros(rect.shape)
    try:
        coordinates = mesh.axes[index[0]].cartesian_coordinates(mesh.axes[index[1]])
    except (NotImplementedError, TypeError):
        coordinates = mesh.axes[index[1]].cartesian_coordinates(mesh.axes[index[0]]).transpose()
    x = coordinates[0][:, 0]
    y = coordinates[1][0]
    coord = [x, y]
    for i, row in enumerate(mask):
        for j, _ in enumerate(row):
            cell_coord = [coord[0][i], coord[1][j]]
            mask[i, j] = 1 - max(np.abs((cell_coord[0] - center[0]) / (size[0])),
                                 np.abs((cell_coord[1] - center[1]) / (size[1]))) * 2
    mask = mask.clip(min=0)
    res = rect * mask
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)
    return res


def cone(mesh, center=(3, 4), ax_len=(4, 3), index=(0, 1), height=1, cone_type='cone', resolution=32, broadcast=True):
    """Create solution array, representing  2d real_solution defined by specified parameters
        with density changing as height of the elliptical cone .

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        center (tuple of int, optional): Center of the real_solution, given by tuples with 2 elements(x, y).
            Default: (0, 0).
        ax_len (tuple of int, optional): Half-width and Half-height of the base real_solution,
            given by tuples with 2 elements (a, b). default: (5, 5).
        index (tuple of two ints, optional): axes to build object at. Default:  (0,1)
        height (float, optional): Cone max height. Minimum height is 0. default: 1
        cone_type (str, {'cone', 'paraboloid', 'paraboloid_h'}, optional): Shape of cone.
        resolution (integer, optional): Relative number of points, approximating base real_solution. default: 32.
        broadcast (bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
             If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: 2D numpy array, representing cone on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
        TypeError if cone type is unknown.
    """
    for i in index:
        if mesh.axes[i] is not cartesian.Axis1d:
            raise TypeError("such combination of axes is not supported.")
    ell = ellipse(mesh, center, ax_len, index, height, resolution, broadcast=False)
    mask = np.zeros(ell.shape)
    try:
        coordinates = mesh.axes[index[0]].cartesian_coordinates(mesh.axes[index[1]])
    except (NotImplementedError, TypeError):
        coordinates = mesh.axes[index[1]].cartesian_coordinates(mesh.axes[index[0]]).transpose()
    x = coordinates[0][:, 0]
    y = coordinates[1][0]
    coord = [x, y]
    for i, row in enumerate(mask):
        for j, _ in enumerate(row):
            cell_coord = [coord[0][i], coord[1][j]]
            if cone_type == 'cone':
                k = np.sqrt(1 / (np.abs((cell_coord[0] - center[0]) ** 2 / (ax_len[0]))
                                 + np.abs((cell_coord[1] - center[1]) ** 2 / (ax_len[1]))))
                if k == 0:
                    mask[i, j] = 1
                else:
                    h = (k - 1) / k
                    mask[i, j] = h
            elif cone_type == 'paraboloid_h':
                mask[i, j] = 1 - np.abs((cell_coord[0] - center[0]) ** 2 / (ax_len[0])) + np.abs(
                    (cell_coord[1] - center[1]) ** 2 / (ax_len[1]))
            elif cone_type == 'paraboloid':
                mask[i, j] = 1 - np.sqrt(np.abs((cell_coord[0] - center[0]) / (ax_len[0])) ** 2 + np.abs(
                    (cell_coord[1] - center[1]) / (ax_len[1])) ** 2)
            else:
                raise TypeError("Unknown cone type. Correct types are 'cone', 'paraboloid', 'paraboloid_h'.")
    mask = mask.clip(min=0)
    res = ell * mask
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)
    return res
