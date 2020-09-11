"""Routines to work with geometry
"""
import numpy as np
import shapely.geometry
from . import array_routines
from tomomak.mesh import cartesian


def intersection_2d(mesh, points, index=(0, 1), calc_area=True):
    """Create array, representing 2d polygon, defined by specified points on the given mesh.

    Each value in the array corresponds to the intersection area of the given cell and the polygon.
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
        index (tuple of one or two ints, optional): axes to build object at. Default:  (0,1)
        calc_area (bool): If True, area of intersection with each cell is calculated, if False,
            only fact of intersecting with mesh cell is taken into account. Default: True.

    Returns:
        ndarray: 1D or 2D numpy array, representing polygon on the given mesh.

    Raises:
        TypeError if no combination of axes supports the cell_edges2d_cartesian() method or all axes dimensions are > 2.
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    pol = shapely.geometry.Polygon(points)
    # If axis is 2D
    if mesh.axes[index[0]].dimension == 2:
        i1 = index[0]
        try:
            cells = mesh.axes[i1].cell_edges2d_cartesian()
            shape = (mesh.axes[i1].size,)
            res = np.zeros(shape)
            for i, row in enumerate(res):
                cell = shapely.geometry.Polygon(cells[i])
                if calc_area:
                    if pol.intersects(cell):
                        res[i] = pol.intersection(cell).area
                else:
                    inters = pol.intersects(cell)
                    if inters:
                        res[i] = cell.area
            return res
        except (TypeError, AttributeError) as e:
            raise type(e)(e.message + "Custom axis should implement cell_edges2d_cartesian method. "
                                      "This method returns 2d list of ordered sequence of point tuples."
                                      " See docstring for more information.")
    # If axes are 1D
    elif mesh.axes[0].dimension == 1:
        i1 = index[0]
        i2 = index[1]
        try:
            cells = mesh.axes[i1].cell_edges2d_cartesian(mesh.axes[i2])
        except (TypeError, AttributeError, NotImplementedError):
            try:
                cells = np.array(mesh.axes[i2].cell_edges2d_cartesian(mesh.axes[i1]), dtype=object)
                cells =np.transpose(cells)
            except (NotImplementedError, TypeError) as e:
                raise type(e)("Custom axis should implement cell_edges2d_cartesian method. "
                              "This method returns 2d list of ordered sequence of point tuples."
                              " See docstring for more information.")
    else:
        raise TypeError("2D objects can be built on the 1D and 2D axes only.")
    shape = (mesh.axes[i1].size, mesh.axes[i2].size)
    res = np.zeros(shape)
    for i, row in enumerate(res):
        for j, _ in enumerate(row):
            cell = shapely.geometry.Polygon(cells[i][j])
            if calc_area:
                if pol.intersects(cell):
                    res[i, j] = pol.intersection(cell).area
            else:
                inters = pol.intersects(cell)
                if inters:
                    res[i, j] = 1
    return res


def cell_areas(mesh, index):
    """Get area of each cell on 2D mesh.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints, optional): axes to build object at. Default:  (0,1)

    Returns:
        ndarray: 2D or 1D ndarray with cell areas.
    """
    check_spatial(mesh, index)
    # If axis is 2D
    if mesh.axes[index[0]].dimension == 2:
        i1 = index[0]
        shape = (mesh.axes[i1].size,)
        ds = np.zeros(shape)
        cells = mesh.axes[i1].cell_edges2d_cartesian()
        for i, row in enumerate(ds):
            cell = shapely.geometry.Polygon(cells[i])
            ds[i] = cell.area
    # If axes are 1D
    elif mesh.axes[0].dimension == 1:
        i1 = index[0]
        i2 = index[1]
        shape = (mesh.axes[i1].size, mesh.axes[i2].size)
        ds = np.zeros(shape)
        try:
            cells = mesh.axes[i1].cell_edges2d_cartesian(mesh.axes[i2])
        except (TypeError, AttributeError):
            cells = np.transpose(np.array(mesh.axes[i2].cell_edges2d_cartesian(mesh.axes[i1]), dtype=object))
        for i, row in enumerate(ds):
            for j, _ in enumerate(row):
                cell = shapely.geometry.Polygon(cells[i][j])
                ds[i, j] = cell.area
    return ds


def cell_distances(mesh, index, p):
    """Get distance to each cell on 2D mesh.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of two ints, optional): axes to calculate distance at. Default:  (0,1)
        p (list of two floats): list representing point in 2D coordinates.

    Returns:
        ndarray: 2D or 1D ndarray with distances.

    Raises
        TypeError if axis dimension is > 2.
        """
    check_spatial(mesh, index)
    p1 = shapely.geometry.Point(p)
    # If axis is 2D
    if mesh.axes[index[0]].dimension == 2:
        i1 = index[0]
        shape = (mesh.axes[i1].size,)
        r = np.zeros(shape)
        coordinates = mesh.axes[i1].cartesian_coordinates()
        for i, row in enumerate(r):
            p2 = shapely.geometry.Point(coordinates[0][i], coordinates[1][i])
            r[i] = p1.distance(p2)
    # If axes are 1D
    elif mesh.axes[0].dimension == 1:
        i1 = index[0]
        i2 = index[1]
        shape = (mesh.axes[i1].size, mesh.axes[i2].size)
        r = np.zeros(shape)
        try:
            coordinates = mesh.axes[i1].cartesian_coordinates(mesh.axes[i2])
        except (NotImplementedError, TypeError):
            coordinates = mesh.axes[i2].cartesian_coordinates(mesh.axes[i1])
            coordinates = (coordinates[0].transpose(), coordinates[1].transpose())
        for i, row in enumerate(r):
            for j, _ in enumerate(row):
                p2 = shapely.geometry.Point(coordinates[0][i, j], coordinates[1][i, j])
                r[i, j] = p1.distance(p2)
    else:
        raise TypeError("Only 1D and 2D axes may be used for 2D distance calculation.")
    return r


def check_spatial(mesh, index):
    """Check, that all axes are spatial.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        index (tuple of ints): axes to check at.

    Returns:
        None

    Raises:
        ValueError if one or more axes are not spatial.
    """
    for i in index:
        if not mesh.axes[i].spatial:
            raise ValueError('Only spatial axes may be used with geometry modules.')


def _convert_slice_cart(data, mesh, index, direct):
    if isinstance(index, int):
        index = [index]
    if len(index) != len(data.shape):
        raise ValueError('index and data have different sizes')
    axes = []
    for i in index:
        axes.append((mesh.axes[i]))
    if any([type(a) is not cartesian.Axis1d for a in axes]):
        # If axis is 2D
        if mesh.axes[index[0]].dimension == 2:
            areas = cell_areas(mesh, index)
            dv = 1 / mesh.axes[index[0]].volumes
            areas *= dv
        # If axes are 1D
        elif mesh.axes[0].dimension == 1:
            areas = cell_areas(mesh, index)
            for i in range(2):
                dv = 1 / mesh.axes[index[i]].volumes
                areas = array_routines.multiply_along_axis(areas, dv, i)
        else:
            raise ValueError('Incorrect axes dimensions')
        if direct:
            new_data = data / areas
        else:
            new_data = data * areas
    else:
        return data
    return new_data


def convert_slice_from_cartesian(data, mesh, index, data_type):
    """Converts 2D slice in cartesian coordinates to mesh coordinates.

    Args:
        data (numpy array): 2D slice of solution or detector geometry.
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
    """Converts 2D slice in  mesh coordinates to cartesian coordinates.

    Args:
        data (numpy array): 2D slice of solution or detector geometry.
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
        # If axis is 2D
        if mesh.axes[index[0]].dimension == 2:
            c_areas = cell_areas(mesh, index)
            dv = 1 / mesh.axes[index[0]].volumes
            c_areas = c_areas * dv
            new_data = np.moveaxis(data, index, (-1,))
            new_data = new_data * c_areas
            new_data = np.moveaxis(new_data, (-1,), index)
        # If axes are 1D
        elif mesh.axes[0].dimension == 1:
            c_areas = cell_areas(mesh, index)
            for i in range(2):
                dv = 1 / mesh.axes[index[i]].volumes
                c_areas = array_routines.multiply_along_axis(c_areas, dv, i)
            new_data = np.moveaxis(data, index, (-2, -1))
            if direct:
                new_data = new_data / c_areas
            else:
                new_data = new_data * c_areas
            new_data = np.moveaxis(new_data, (-2, -1), index)
        else:
            raise ValueError('Incorrect axes dimensions')
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


