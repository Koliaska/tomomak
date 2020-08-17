import tomomak.util.geometry3d as geometry3d
import tomomak.util.array_routines as array_routines
import numpy as np


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
              det_num (integer): number of detectors.
              *args: fan_detector arguments.
              **kwargs: fan_detector keyword arguments.

          Returns:
              ndarray: numpy array, representing fan of detectors on a given mesh.
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