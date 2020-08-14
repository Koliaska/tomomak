import numpy as np
from scipy.interpolate import LinearNDInterpolator
import itertools
import trimesh
import os
import subprocess
import sys


def get_trimesh(mesh, index=(0, 1, 2)):
    """Get array of trimesh objects (3D meshes) corresponding to each grid cell.

    Helper function. Trimesh objects are used for complex 3d calculations.
    Only axes which implements cell_edges3d method are supported (see abstract_axes.py).
    Trimesh module is used for the calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.:
        index(tuple of one, two or three ints, optional): axes to build object at. Default:  (0, 1, 2)

    Returns:
        list: 1D/2D or 3D list (depending on used tomomak mesh), containing trimesh representation of each cell.

    Raises:
        TypeError if no combination of axes supports the cell_edges3d() method or all axes dimensions are > 3.
    """
    if isinstance(index, int):
        index = [index]
    if mesh.axes[index[0]].dimension == 3:
        try:
            (vertices, faces) = mesh.axes[index[0]].cell_edges3d()
            shape = mesh.axes[index[0]].size
            trimesh_list = np.zeros(shape).tolist()
            for i, _ in enumerate(trimesh_list):
                        if faces is None:
                            cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i]))
                        else:
                            cell = trimesh.Trimesh(vertices=vertices[i], faces=faces[i])
                        trimesh_list[i] = cell
        except (TypeError, AttributeError) as e:
            raise type(e)(e.message + "Custom axis should implement cell_edges3d method. "
                                      " See docstring for more information.")
    # If 1st axis is 2D
    elif mesh.axes[index[0]].dimension == 2 or mesh.axes[index[1]].dimension == 2:
        (vertices, faces) = _get_cells2d(index, mesh)
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
        (vertices, faces) = _get_cells3d(index, mesh)
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


def grid_intersection3d(trimesh_list, vertices, faces=None):
    """Create array, representing intersection volume of each object in list  with given object.

    List of object is usually a list, containing each cell of the used 3D grid.
    The object is usually a real-world object or detector line of sight.
    Very slow.

    Args:
        trimesh_list: a list  containing trimesh representation of each cell (e.g. obtained with get_trimesh function).
        vertices: a list of lists of points (x, y, z) in cartesian coordinates: vertices the cell.
        faces: a list of lists of cell faces. Each face is a list of vertices, denoted in cw direction.
            If faces is None, faces are created automatically for the convex hull of the vertices. Default:  None.

    Returns:
        ndarray: numpy array, representing intersection volume of each object in list  with given object.

    Raises:
        TypeError if trimesh_list has 0 or > 3 dimensions.
    """
    if faces is None:
        obj3d = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices))
    else:
        obj3d = trimesh.Trimesh(vertices=vertices, faces=faces)
    list_dim = len(_dim(trimesh_list))
    if list_dim == 3:
        shape = (len(trimesh_list), len(trimesh_list[0]), len(trimesh_list[0][0]))
        res = np.zeros(shape)
        # sys.stderr = open(os.devnull, "w")  # supressing console for trimesh calculations
        for i, row in enumerate(res):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    try:
                        inters = trimesh.boolean.intersection((obj3d, trimesh_list[i][j][k])).volume
                    except subprocess.CalledProcessError:
                        inters = 0
                    res[i, j, k] = inters
        # sys.stderr = sys.__stderr__
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

def _get_cells3d(index, mesh):
    ax = [mesh.axes[index[i]] for i in (0, 1, 2)]
    ind_lst = list(itertools.permutations((0, 1, 2), 3))
    for p in ind_lst:
        try:
            new_axes = [ax[i] for i in p]

            (vertices, faces) = new_axes[0].cell_edges3d(new_axes[1], new_axes[2])
            for i in range(3):
                vertices = np.moveaxis(vertices, p[i], i)
                faces = np.moveaxis(faces, p[i], i)
            return vertices.tolist(), faces.tolist()
        except NotImplementedError:
            pass
    raise TypeError("Custom axis should implement cell_edges3d method. " " See docstring for more information.")


def _get_cells2d(index, mesh):
    try:
        (vertices, faces) = mesh.axes[index[0]].cell_edges3d(mesh.axes[index[1]])
        return vertices.tolist(), faces.tolist()
    except NotImplementedError:
        try:
            (vertices, faces) = mesh.axes[index[1]].cell_edges3d(mesh.axes[index[0]])
            vertices = vertices.transpose()
            faces = faces.transpose()
            return vertices.tolist(), faces.tolist()
        except NotImplementedError:
            raise TypeError("Custom axis should implement cell_edges3d method. " " See docstring for more information.")


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
        locations, index_ray, index_tri = trimesh_element.ray.intersects_location(
            ray_origins=(p1,),
            ray_directions=(p2,))
        if len(locations) > 0:
            inters = trimesh_element.volume
        else:
            inters = 0
    except subprocess.CalledProcessError:
        inters = 0
    return inters

def volumes_3d(mesh, index=(0, 1, 2)):
    """Get array of each cell volumes

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        index(tuple of one, two or three ints, optional): axes to build object at. Default: (0, 1, 2)

    Returns:
        ndarray: numpy array, representing volume of each cell.
    """
    if isinstance(index, int):
        index = [index]
    trimesh_list = get_trimesh(mesh, index)
    volumes = np.zeros_like(trimesh_list)
    for idx, value in np.ndenumerate(trimesh_list):
        volumes[idx] = value.volume
    return volumes


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
    interp = LinearNDInterpolator(points, new_data)
    x_new = np.linspace(x_grid[0, 0, 0], x_grid[-1, -1, -1], interp_size)
    y_new = np.linspace(y_grid[0, 0, 0], y_grid[-1, -1, -1], interp_size)
    z_new = np.linspace(z_grid[0, 0, 0], z_grid[-1, -1, -1], interp_size)
    x_grid, y_grid, z_grid = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    new_points = np.transpose(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
    new_data = interp(new_points)
    new_data = np.reshape(new_data, x_grid.shape)
    return x_grid, y_grid, z_grid, new_data

def show_cell(mesh, index=(0,1,2), cell_index=(0,0,0)):
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
    trimesh_list = get_trimesh(mesh, index)
    for i in cell_index:
        trimesh_list = trimesh_list[i]
    trimesh_list.show()