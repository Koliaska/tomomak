import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .geometry2d import check_spatial


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
