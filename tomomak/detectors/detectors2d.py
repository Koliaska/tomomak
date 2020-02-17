"""Generators for basic detectors and detector arrays in 2D geometry
"""
import shapely.geometry
import shapely.affinity
import tomomak.util.geometry2d
from tomomak.util.engine import muti_proc
from multiprocessing import Pool
import os
import numpy as np


def two_pi_det(mesh, position, index=(0, 1), response=1, radius_dependence=True, broadcast=True):
    """Generate intersection of one 2pi detector (e.g. hermetic detector)  with mesh cells.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        position (tuple of 2 floats): Detector origin (x, y).
        index (tuple of two ints, optional): axes to build object at. Default: (0,1).
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
    areas = tomomak.util.geometry2d.cell_areas(mesh, index)
    intersection_geometry = areas * response
    if radius_dependence:
        r = tomomak.util.geometry2d.cell_distances(mesh, index, position)
        r = 4 * np.pi * np.square(r)
        intersection_geometry /= r
    if broadcast:
        intersection_geometry = tomomak.util.array_routines.broadcast_object(intersection_geometry, index, mesh.shape)
    return intersection_geometry

def two_pi_detector_array(mesh, focus_point, radius, det_num,  *args, **kwargs):
    """ Creates array of fan detectors around focus points.

          Args:
              mesh (tomomak.main_structures.Mesh): mesh to work with.
              focus_point (tuple of 2 floats): Focus point (x, y).
              radius (float): radius of the circle around focus_point, where detectors are located.
              fan_num (integer): number of fans.
              line_num (integer): number of lines.
              width (float): width of each line.
              incline (float): incline of first detector fan in Rad from the (1, 0) direction. Default: 0.
              *args: fan_detector arguments.
              **kwargs: fan_detector keyword arguments.

          Returns:
              ndarray: numpy array, representing fan of detectors on a given mesh.
          """
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    incline = 0
    d_incline = np.pi * 2 / det_num
    focus_point = np.array(focus_point)
    for i in range(det_num):
        p = np.array([focus_point[0] + radius * np.cos(incline), focus_point[1] + radius * np.sin(incline)])
        addition = np.array([two_pi_det(mesh, p,  *args, **kwargs),])
        res = np.append(res, addition, axis=0)
        print('\r', end='')
        print("Generating array of 2pi detectors: ", str(i * 100 // det_num) + "% complete", end='')
        incline += d_incline
    print('\r \r ', end='')
    print('\r \r ', end='')
    return res


def detector2d(mesh, p1, p2, width, divergence=0, index=(0, 1), response=1, radius_dependence=True,
               broadcast=True, calc_area=True):
    """Generate intersection of one detector line with mesh cells.

    Source is isotropic.
    Line length should be long enough to lay outside of mesh.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p1 (tuple of 2 floats): Detector origin (x, y).
        p2 (tuple of 2 floats): Second point, characterizing central axis of detector line.
        width (float): width of line.
        divergence (float, optional): line of sight divergence. 0 means line is collimated. Default: 0.
        index (tuple of two ints, optional): axes to build object at. Default: (0,1).
        response (float, optional): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval. Default: 1.
        radius_dependence (bool, optional): if True, signal is divided by 4 *pi *r^2
        broadcast (bool, optional): If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.
        calc_area (bool, optional): If True, area of intersection with each cell is calculated, if False,
            only fact of intersecting with mesh cell is taken into account. Default: True.

    Returns:
         ndarray: numpy array, representing one detector on a given mesh.
    """
    points = _line_to_polygon(p1, p2, width, divergence)
    if isinstance(index, int):
        index = [index]
    res = tomomak.util.geometry2d.intersection_2d(mesh, points, index, calc_area)
    if radius_dependence:
        r = tomomak.util.geometry2d.cell_distances(mesh, index, p1)
        r = 4 * np.pi * np.square(r)
        res /= r
    res *= response
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)
    return res


def fan_detector(mesh, p1, p2, width, number, index=(0, 1), angle=np.pi/2, *args, **kwargs):
    """ Creates one fan of detectors.

    Args:
        mesh (tomomak.main_structures.Mesh): mesh to work with.
        p1 (tuple of 2 floats): Detector origin (x, y).
        p2 (tuple of 2 floats): Second point, characterizing central axis of detector fan.
        width (folat): width of each line.
        index (tuple of two ints, optional): axes to build object at. Default: (0,1).
        number (integer): number of detector lines in the fan.
        angle (float): total angle of fan in Rad. Default: pi/2.
        *args: line2d arguments.
        **kwargs: line2d keyword arguments.

    Returns:
        ndarray: numpy array, representing fan of detectors on a given mesh.
    """
    if angle < 0 or angle >= np.pi:
        raise ValueError("angle value is {}. It should be >= 0 and < pi.".format(angle))
    # finding first sightline of the detector
    p1 = np.array(p1)
    p2 = np.array(p2)
    if isinstance(index, int):
        index = [index]
    r = p2 - p1
    r = r / np.cos(angle / 2)
    p2 = p1 + r
    line = shapely.geometry.LineString([p1, p2])
    line = shapely.affinity.rotate(line, -angle / 2, origin=p1, use_radians=True)
    rot_angle = angle / (number - 1)
    # start scanning
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    for i in range(number):
        p1, p2 = line.coords
        addition = np.array([detector2d(mesh, p1, p2, width, index=index, *args, **kwargs)])
        res = np.append(res, addition, axis=0)
        line = shapely.affinity.rotate(line, rot_angle, origin=p1, use_radians=True)
    return res


@muti_proc
def fan_detector_array(mesh, focus_point, radius, fan_num, line_num, width,
                       incline=0,  *args, **kwargs):
    """ Creates array of fan detectors around focus points.

    Multiprocess acceleration is supported.
    To turn it on run script with environmental variable TM_MP set to number of desired cores.
    Or just write in your script:
       import os
       os.environ["TM_MP"] = "8"
    If you use Windows, due to Python limitations, you have to guard your script with
    if __name__ == "__main__":
        ...your script

      Args:
          mesh (tomomak.main_structures.Mesh): mesh to work with.
          focus_point (tuple of 2 floats): Focus point (x, y).
          radius (float): radius of the circle around focus_point, where detectors are located.
          fan_num (integer): number of fans.
          line_num (integer): number of lines.
          width (float): width of each line.
          incline (float): incline of first detector fan in Rad from the (1, 0) direction. Default: 0.
          *args: fan_detector arguments.
          **kwargs: fan_detector keyword arguments.

      Returns:
          ndarray: numpy array, representing fan of detectors on a given mesh.
      """
    pass


def _fan_detector_array(mesh, focus_point, radius, fan_num, line_num, width,
                        incline=0,  *args, **kwargs):
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    fans = _prepare_fans(focus_point, radius, fan_num, incline)
    for i, f in enumerate(fans):
        res = np.append(res, fan_detector(mesh, f[0], f[1], width, line_num,  *args, **kwargs), axis=0)
        print('\r', end='')
        print("Generating array of fan detectors: ", str(i*100 // fan_num) + "% complete", end='')
    print('\r \r ', end='')
    print('\r \r ', end='')
    return res


def _prepare_fans(focus_point, radius, fan_num, incline):
    d_incline = np.pi * 2 / fan_num
    focus_point = np.array(focus_point)
    res = []
    for i in range(fan_num):
        p1 = np.array([focus_point[0] + radius * np.cos(incline), focus_point[1] + radius * np.sin(incline)])
        r = (focus_point - p1) * 10
        p2 = p1 + r
        res.append([p1, p2])
        incline += d_incline
    return res


def _fan_detector_array_mp(mesh, focus_point, radius, fan_num, line_num, width,
                           incline=0, *args, **kwargs):
    proc_num = int(os.getenv('TM_MP'))
    pool = Pool(processes=proc_num)
    print("Started multi-process calculation of 2D fan detector array on {} cores.".format(proc_num))
    res = []
    fans = _prepare_fans(focus_point, radius, fan_num, incline)
    for i, f in enumerate(fans):
        res.append(pool.apply_async(fan_detector, (mesh, f[0], f[1], width, line_num) + args, kwargs))
    pool.close()
    pool.join()
    shape = [0]
    shape.extend(mesh.shape)
    final_res = np.zeros(shape)
    for r in res:
        final_res = np.append(final_res, r.get(), axis=0)
    return final_res


def parallel_detector(mesh, p1, p2, width, number, shift, index=(0, 1), *args, **kwargs):
    """ Creates array of parallel detectors.

       Args:
           mesh (tomomak.main_structures.Mesh): mesh to work with.
           p1 (tuple of 2 floats): Detector origin (x, y).
           p2 (tuple of 2 floats): Second point, characterizing central axis of detectors.
           width (float): width of each line.
           number (int): number of detectors.
           shift (float): shift of each line as compared to previous.
           index (tuple of two ints, optional): axes to build object at. Default: (0,1).
           *args: line2d arguments.
           **kwargs: line2d keyword arguments.

       Returns:
           ndarray: numpy array, representing detectors on a given mesh.
       """
    # finding first sightline of the detector
    p1 = np.array(p1)
    p2 = np.array(p2)
    if isinstance(index, int):
        index = [index]
    r = p2 - p1
    r = r * 5
    p2 = p1 + r
    line = shapely.geometry.LineString([p1, p2])
    # start scanning
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    for i in range(number):
        p1, p2 = line.coords
        addition = np.array([detector2d(mesh, p1, p2, width, index=index, *args, **kwargs)])
        res = np.append(res, addition, axis=0)
        line = line.parallel_offset(shift, 'left')
    return res


def _line_to_polygon(p1, p2, width, divergence=0):
    """Generate detector geometry for one Line of Sight.

    line of sight can be collimated or diverging (cone-like).

    Args:
        p1 (tuple of floats): first line point (x,y).
        p2 (tuple of 2 points): second line point (x,y).
        width (float): line width.
        divergence (float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
            default: 0.

    Returns:
        tuple of 4 points: List of 4 Shapely points, defining LOS.

    Raises:
        ValueError if divergence is < 0 or >= pi.
    """
    if divergence < 0 or divergence >= np.pi:
        raise ValueError("Divergence value is {}. It should be >= 0 and < pi.".format(divergence))
    # increase line length in case of line rotation
    p1 = np.array(p1)
    p2 = np.array(p2)
    r = p2 - p1
    r = r / np.cos(divergence)
    p2 = p1 + r
    # take into account width and divergence
    line = shapely.geometry.LineString([p1, p2])
    ll = line.parallel_offset(width/2, 'left')
    p_top = ll.coords[0]
    ll = shapely.affinity.rotate(ll, divergence/2, origin=p_top, use_radians=True)
    lr = line.parallel_offset(width/2, 'right')
    p_top = lr.coords[1]
    lr = shapely.affinity.rotate(lr, -divergence/2, origin=p_top, use_radians=True)
    p1, p2 = ll.coords
    p3, p4 = lr.coords
    return p1, p2, p3, p4
