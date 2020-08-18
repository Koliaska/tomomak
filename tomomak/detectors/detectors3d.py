import tomomak.util.geometry3d as geometry3d
import tomomak.util.array_routines as array_routines
import numpy as np
import trimesh


def four_pi_det(mesh, position, index=(0, 1, 2), response=1, radius_dependence=True, broadcast=True):
    """Generate intersection of one 4pi detector (e.g. hermetic detector)  with mesh cells.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        position (tuple of 3 floats): Detector origin (x, y, z).
        index (tuple of 3 ints, optional): axes to build object at. Default: (0,1, 2).
        response (float, optional): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
        radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2
        broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
         ndarray: numpy array, representing one detector on a given mesh.
    """
    if isinstance(index, int):
        index = [index]
    volumes = geometry3d.cell_volumes(mesh, index)
    volumes = volumes * response
    if radius_dependence:
        distances = geometry3d.cell_distances(mesh, position, index)
        # zero = np.argwhere(distances == 0)
        # if zero.size:
        #     sorted_dist = np.sort(distances, axis=None)
        #     distances[tuple(zero[0])] = sorted_dist[1] / (2 ** 4 / 3)
        volumes /= 4 * np.pi * distances ** 2
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def four_pi_detector_array(mesh, focus_point, radius, theta_num, phi_num, index=(0, 1, 2),
                           response=1, radius_dependence=True, broadcast=True):
    """ Creates array of 4pi detectors equally spaced around the focus points.
          Args:
              mesh (tomomak.main_structures.Mesh): mesh to work with.
              focus_point (tuple of 2 floats): Focus point (x, y).
              radius (float): radius of the circle around focus_point, where detectors are located.
              theta_num (int): number of theta steps in spherical coordinates.
              phi_num (int): number of phi steps in spherical coordinates.
              index (tuple of 3 ints, optional): axes to build object at. Default: (0,1, 2).
              response (float, optional): Detector response = amplification * detector area.
                E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
              radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2
              broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
                If False, 2d array is returned, even if Mesh is not 2D. Default: True.
          Returns:
              ndarray: numpy array, representing group of 4pi detectors on a given mesh.
          """
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    volumes = geometry3d.cell_volumes(mesh, index)
    volumes = volumes * response
    for x_n in range(theta_num):
        theta = np.pi / (theta_num + 2) * (x_n + 1)
        for y_n in range(phi_num):
            phi = 2 * np.pi / phi_num * y_n
            x = focus_point[0] + radius * np.sin(theta) * np.cos(phi)
            y = focus_point[1] + radius * np.sin(theta) * np.sin(phi)
            z = focus_point[2] + radius * np.cos(theta)
            position = (x, y, z)
            v = volumes * 1
            if radius_dependence:
                distances = geometry3d.cell_distances(mesh, position, index)
                v /= 4 * np.pi * distances ** 2
            if broadcast:
                v = array_routines.broadcast_object(v, index, mesh.shape)
            addition = np.array([v, ])
            res = np.append(res, addition, axis=0)
        print('\r', end='')
        print("Generating array of 4pi detectors: ", str(x_n * 100 // theta_num) + "% complete", end='')
    print('\r \r', end='')
    print('\r \r', end='')
    return res


def line_detector(mesh, p1, p2, radius, calc_volume, index=(0, 1, 2),
                  response=1, radius_dependence=True, broadcast=True):
    """Generate intersection of detector with ray-like line of sight and mesh cells.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p1 (tuple of 3 floats): Detector origin (x, y, z).:
        p2 (tuple of 3 floats): Detector line of sight direction(x, y, z).
        radius (float): Line of sight radius. If calc_volume is True, radius should be set to None.
        calc_volume (bool): If true, volume of intersection of each cell and detector line of sight is calculated. Slow.
            If false, only fact of intersection of line of sight ray and cell is taken into account.
            If line of sight intersects the cell, returned array has 1 at corresponding position.
        index (tuple of 3 ints, optional): axes to build object at. Default: (0,1, 2).
        response (float, optional): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
        radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2
        broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: numpy array, representing one detector on a given mesh.

    Raises:
        AttributeError if excess parameter radius is not None when calc_volume = False.
    """
    if isinstance(index, int):
        index = [index]
    trimesh_list = geometry3d.get_trimesh_grid(mesh, index)
    distances = None
    if calc_volume:
        distances = geometry3d.cell_distances(mesh, p1, index)
        max_dist = np.max(distances) * 1.2
        v1 = np.array((0, 0, -1))
        v2 = np.array(p2) - np.array(p1)
        rot_matr = _rotation_matrix_from_vectors(v1, v2)
        rot_vector = np.dot(rot_matr, v1) * max_dist / 2
        rot_vector += np.array(p1)
        shift = np.array([rot_vector, ]).transpose()
        footer = np.array([[0, 0, 0, 1]])
        transform_matrix = np.append(rot_matr, shift, axis=1)
        transform_matrix = np.append(transform_matrix, footer, axis=0)
        obj3d = trimesh.creation.cylinder(radius, max_dist)
        obj3d.apply_transform(transform_matrix)
        volumes = geometry3d.grid_intersection3d(trimesh_list, obj3d)
    else:
        if radius is not None:
            raise AttributeError("Radius is not used when calc_volume is False. Set radius to None.")
        volumes = geometry3d.grid_ray_intersection(trimesh_list, p1, p2)
    volumes *= response
    if radius_dependence:
        if distances is None:
            distances = geometry3d.cell_distances(mesh, p1, index)
        volumes /= 4 * np.pi * distances ** 2
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def _rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2

    From stackoverflow
    Args:
        vec1: A 3d "source" vector
        vec2: A3d "destination" vector

    Returns: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
