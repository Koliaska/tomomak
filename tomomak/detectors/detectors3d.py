import tomomak.util.geometry3d_basic as geometry3d
import tomomak.util.array_routines as array_routines
import tomomak.util.geometry3d_trimesh
from tomomak.util.engine import muti_proc
from tomomak.util import text
import numpy as np
try:
    import trimesh
except ImportError:
    pass
from multiprocessing import Pool
import os


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
    volumes = tomomak.util.geometry3d_trimesh.cell_volumes(mesh, index)
    volumes = volumes * response
    if radius_dependence:
        distances = geometry3d.cell_distances(mesh, position, index)
        # zero = np.argwhere(distances == 0)
        # if zero.size:
        #     sorted_dist = np.sort(distances, axis=None)
        #     distances[tuple(zero[0])] = sorted_dist[1] / (2 ** 4 / 3)
        volumes /= 4 * np.pi * distances ** 2
    volumes = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(volumes, mesh, index, data_type='detector_geometry')
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def four_pi_detector_array(mesh, focus_point, radius, theta_num, phi_num, index=(0, 1, 2),
                           response=1, radius_dependence=True, broadcast=True):
    """ Creates array of 4pi detectors equally spaced by angles around the focus points.

        Note that detectors are evenly spaced by angles in spherical coordinates,
         but not uniformly distributed around the sphere.
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
    volumes = tomomak.util.geometry3d_trimesh.cell_volumes(mesh, index)
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
            v = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(v, mesh, index, data_type='detector_geometry')
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
        p2 (tuple of 3 floats): Another point characterizing detector line of sight direction (p1->p2).
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
        ValueError if excess parameter radius is not None when calc_volume = False.
    """
    if isinstance(index, int):
        index = [index]
    trimesh_list = tomomak.util.geometry3d_trimesh.get_grid(mesh, index)
    distances = None
    if calc_volume:
        distances = geometry3d.cell_distances(mesh, p1, index)
        max_dist = np.max(distances) * 1.3
        transform_matrix = tomomak.util.geometry3d_trimesh.transform_matrix(p1, p2, max_dist / 2)
        obj3d = trimesh.creation.cylinder(radius, max_dist)
        obj3d.apply_transform(transform_matrix)
        volumes = tomomak.util.geometry3d_trimesh.grid_intersection3d(trimesh_list, obj3d)
    else:
        if radius is not None:
            raise ValueError("Radius is not used when calc_volume is False. Set radius to None.")
        volumes = tomomak.util.geometry3d_trimesh.grid_ray_intersection(trimesh_list, p1, p2)
    volumes *= response
    if radius_dependence:
        if distances is None:
            distances = geometry3d.cell_distances(mesh, p1, index)
        volumes /= 4 * np.pi * distances ** 2
    volumes = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(volumes, mesh, index, data_type='detector_geometry')
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def cone_detector(mesh, p1, p2, divergence, index=(0, 1, 2),
                  response=1, radius_dependence=True, broadcast=True):
    """Generate intersection of detector with cone-like line of sight and mesh cells.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p1 (tuple of 3 floats): Detector origin (x, y, z).:
        p2 (tuple of 3 floats): Another point characterizing detector line of sight direction (p1->p2).
        divergence (float): Cone divergence in radians.
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
    trimesh_list = tomomak.util.geometry3d_trimesh.get_grid(mesh, index)
    distances = geometry3d.cell_distances(mesh, p1, index)
    max_dist = np.max(distances) * 1.3 / np.cos(divergence / 2)
    transform_matrix = tomomak.util.geometry3d_trimesh.transform_matrix(p1, p2, max_dist)
    radius = max_dist * np.sin(divergence)
    obj3d = trimesh.creation.cone(radius, max_dist)
    obj3d.apply_transform(transform_matrix)
    volumes = tomomak.util.geometry3d_trimesh.grid_intersection3d(trimesh_list, obj3d)
    volumes *= response
    if radius_dependence:
        volumes /= 4 * np.pi * distances ** 2
    volumes = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(volumes, mesh, index, data_type='detector_geometry')
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def custom_detector(mesh, vertices, detector_origin=None, index=(0, 1, 2),
                    response=1, radius_dependence=True, broadcast=True):
    """ Generate intersection of detector with line of sight defined by given vertices and mesh cells.

    Note that for a typical detector all vertices should lay outside of the mesh.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        vertices (array-like): a list of lists of points (x, y, z) in cartesian coordinates,
            characterizing detector line of sight.
        detector_origin (tuple of 3 ints, optional): If detector_origin is defined,
            distance to this point is calculated for radius_dependence.
            Otherwise distance to first vertices is calculated.
        index (tuple of 3 ints, optional): axes to build object at. Default: (0,1, 2).
        response (float, optional): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
        radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2.
        broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: numpy array, representing one detector on a given mesh.
    """
    if isinstance(index, int):
        index = [index]
    obj3d = tomomak.util.geometry3d_trimesh.get_obj(vertices)
    trimesh_list = tomomak.util.geometry3d_trimesh.get_grid(mesh, index)
    volumes = tomomak.util.geometry3d_trimesh.grid_intersection3d(trimesh_list, obj3d)
    volumes *= response
    if radius_dependence:
        if detector_origin is None:
            detector_origin = vertices[0]
        distances = geometry3d.cell_distances(mesh, detector_origin, index)
        volumes /= 4 * np.pi * distances ** 2
    volumes = tomomak.util.geometry3d_trimesh.convert_slice_from_cartesian(volumes, mesh, index, data_type='detector_geometry')
    if broadcast:
        volumes = array_routines.broadcast_object(volumes, index, mesh.shape)
    return volumes


def aperture_detector(mesh, detector_vertices, aperture_vertices, detector_origin=None, index=(0, 1, 2),
                      response=1, radius_dependence=True, broadcast=True):
    """Generate intersection of detector with line of sight, defined by detector and aperture planes, and mesh cells.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        detector_vertices (array-like): a list of lists of points (x, y, z) in cartesian coordinates,
            characterizing detector plane.
        aperture_vertices(array-like): a list of lists of points (x, y, z) in cartesian coordinates,
            characterizing aperture plane.
        detector_origin (tuple of 3 ints, optional): If detector_origin is defined,
            distance to this point is calculated for radius_dependence.
            Otherwise distance to the mean of detector_vertices is calculated.
        index (tuple of 3 ints, optional): axes to build object at. Default: (0, 1, 2).
        response (float, optional): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
        radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2.
        broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: numpy array, representing one detector on a given mesh.
    """
    if isinstance(index, int):
        index = [index]
    if detector_origin is None:
        detector_origin = np.mean(detector_vertices, axis=0)

    distances = geometry3d.cell_distances(mesh, detector_origin, index)
    mesh_center = np.array(geometry3d.mesh_center(mesh, index))
    max_dist = np.max(distances) * 2
    ver_list = []
    detector_vertices = np.asarray(detector_vertices)
    aperture_vertices = np.array(aperture_vertices)
    # generate points for line of sight
    for det_p in detector_vertices:
        ver_list.append(det_p)
        for ap_p in aperture_vertices:
            direction = (ap_p - det_p) / np.linalg.norm((ap_p - det_p))
            center_direction = (mesh_center - ap_p) / np.linalg.norm((mesh_center - ap_p))
            cos = np.dot(direction, center_direction) / (np.linalg.norm(direction) * np.linalg.norm(center_direction))
            ver_list.append(det_p + direction * max_dist / np.abs(cos))
    return custom_detector(mesh, ver_list, detector_origin, index, response, radius_dependence, broadcast)


@muti_proc
def fan_detector(mesh, p0, boundary_points, number, det_type='cone', index=(0, 1, 2), *args, **kwargs):
    """ Generate 1d or d2 fan of line or cone detectors.

    For 2d fan multiprocess acceleration is supported.
    To turn it on run script with environmental variable TM_MP set to number of desired cores.
    Or just write in your script:
       import os
       os.environ["TM_MP"] = "8"
    If you use Windows, due to Python limitations, you have to guard your script with
    if __name__ == "__main__":
        ...your script

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p0 (tuple of 3 floats): Detector origin (x, y, z).:: Detector origin (x, y, z).
        boundary_points (tuple of 2 or four points): points, defining boundary detector line of sights.
         If 2 points are defined, 1d fan is generated. Detector lines are evenly spaced between (p0->boundary_points[0])
         and (p0->boundary_points[1]) vectors.
         If 4 points are defined, 2d fan is generated.
         Detector lines are evenly spaced between the four planes, defined by p0 and four boundary points.
        number (int or tuple of 2 ints): number of detectors for 1d fan. Or number of columns and rows in 2d fan.
        det_type (str): Detector type. Available options: 'cone', 'line'. Default: 'cone'.
        index (tuple of 3 ints, optional): axes to build object at. Default: (0,1, 2).
        *args: passed to cone_detector or line_detector functions.
        **kwargs: passed to cone_detector or line_detector functions.

    Returns:
        ndarray: numpy array, representing array of detectors on a given mesh.
    """
    pass


def _fan_detector(mesh, p0, boundary_points, number, det_type='cone', index=(0, 1, 2), *args, **kwargs):
    p0 = np.array(p0)
    boundary_points = np.array(boundary_points)
    detectors = []
    if boundary_points.shape[0] == 2:
        detectors = _generate_detector_line(number, boundary_points[0], boundary_points[1],
                                            p0, detectors, det_type, mesh, index, *args, **kwargs)
    elif boundary_points.shape[0] == 4:
        col_num = number[0]
        row_num = number[1]
        col_step1 = (boundary_points[2] - boundary_points[0]) / (col_num - 1)
        col_step2 = (boundary_points[3] - boundary_points[1]) / (col_num - 1)
        print("Generating array of fan detectors: 0% complete", end='')
        for j in range(col_num):
            points = np.array([[boundary_points[0] + col_step1 * j], [boundary_points[1] + col_step2 * j]])
            det = _generate_detector_line(row_num, points[0], points[1],
                                          p0, det_type, mesh, index, *args, **kwargs)
            detectors.extend(det)
            print('\r', end='')
            print("Generating array of fan detectors: ", str(j * 100 // col_num) + "% complete", end='')
        print('\r \r', end='')
        print('\r \r', end='')
    else:
        raise ValueError('Array of boundary_points should contain 2 or 4 points.')
    return np.array(detectors)


def _generate_detector_line(n, bp1, bp2, p0, det_type, mesh, index, *args, **kwargs):
    step = (bp2 - bp1) / (n - 1)
    detectors = []
    for i in range(n):
        p2 = bp1 + step * i
        if det_type == 'cone':
            det = cone_detector(mesh, p0, p2, index=index, *args, **kwargs)
        elif det_type == 'line':
            det = line_detector(mesh, p0, p2, index=index, *args, **kwargs)
        else:
            raise ValueError('detector type {} is unknown'.format(det_type))
        detectors.append(det)
    return detectors


def _fan_detector_mp(mesh, p0, boundary_points, number, det_type='cone', index=(0, 1, 2), *args, **kwargs):
    p0 = np.array(p0)
    boundary_points = np.array(boundary_points)
    detectors = []
    if boundary_points.shape[0] == 2:
        detectors = _generate_detector_line(number, boundary_points[0], boundary_points[1],
                                            p0, detectors, det_type, mesh, index, *args, **kwargs)
    elif boundary_points.shape[0] == 4:
        proc_num = int(os.getenv('TM_MP'))
        pool = Pool(processes=proc_num)
        res = []
        print("Started multi-process calculation of 2D fan detector array on {} cores.".format(proc_num))
        col_num = number[0]
        row_num = number[1]
        col_step1 = (boundary_points[2] - boundary_points[0]) / (col_num - 1)
        col_step2 = (boundary_points[3] - boundary_points[1]) / (col_num - 1)
        for j in range(col_num):
            points = np.array([[boundary_points[0] + col_step1 * j], [boundary_points[1] + col_step2 * j]])
            res.append(pool.apply_async(_generate_detector_line, (row_num, points[0], points[1],
                                                                  p0, det_type, mesh, index) + args, kwargs))
        text.progress_mp(res, col_num)
        pool.close()
        pool.join()
        shape = [0]
        shape.extend(mesh.shape)
        detectors = np.zeros(shape)
        for r in res:
            detectors = np.append(detectors, r.get(), axis=0)
    else:
        raise ValueError('Array of boundary_points should contain 2 or 4 points.')
    return np.array(detectors)
