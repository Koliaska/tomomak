"""Generators for basic detectors arrays in 2D geometry
"""
import shapely.geometry
import shapely.affinity
import tomomak.util.geometry2d
import numpy as np


def line2d(mesh, p1, p2, width, divergence=0, index=(0, 1), response=1, radius_dependence=True,
           broadcast=True, calc_area=True):
    """Generate intersection with one detector line with given parameters.

    Source is isotropic.
    Line length should be long enough to lay outside of mesh.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        p1(tuple of 2 floats): Detector origin (x, y).
        p2(tuple of 2 floats): Second point, characterizing central axis of detector line.
        width: width of line.
        divergence: line of sight divergence. 0 means line is collimated. Default: 0.
        index(tuple of two ints, optional): axes to build object at. Default: (0,1).
        response(float): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval.
        radius_dependence: if True, signal is divided by 4 *pi *r^2
        broadcast(bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.
        calc_area(bool): If True, area of intersection with each cell is calculated, if False,
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
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        p1(tuple of 2 floats): Detector origin (x, y).
        p2(tuple of 2 floats): Second point, characterizing central axis of detector fan.
        width: width of each line.
        index(tuple of two ints, optional): axes to build object at. Default: (0,1).
        number(integer): number of detector lines in the fan.
        angle(float): total angle of fan in Rad. Default: pi/2.
        *args, **kwarg - line2d arguments.

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
        addition = np.array([line2d(mesh, p1, p2, width, index=index,  *args, **kwargs)])
        res = np.append(res, addition, axis=0)
        line = shapely.affinity.rotate(line, rot_angle, origin=p1, use_radians=True)
    return res


def fan_detector_array(mesh, focus_point, radius, fan_num, line_num, width,
                       incline=0,  *args, **kwargs):
    """ Creates array of fan detectors around focus points.

      Args:
          mesh(tomomak.main_structures.Mesh): mesh to work with.
          focus_point(tuple of 2 floats): Focus point (x, y).
          radius(float): radius of the circle around focus_point, where detectors are located.
          fan_num(integer): number of fans.
          line_num(integer): number of lines.
          width: width of each line.
          incline(float): incline of first detector fan in Rad from the (1, 0) direction. Default: 0.
          *args, **kwarg - fan_detector arguments.

      Returns:
          ndarray: numpy array, representing fan of detectors on a given mesh.
      """
    shape = [0]
    shape.extend(mesh.shape)
    res = np.zeros(shape)
    d_incline = np.pi * 2 / fan_num
    focus_point = np.array(focus_point)
    for i in range(fan_num):
        p1 = np.array([focus_point[0] + radius * np.cos(incline), focus_point[1] + radius * np.sin(incline)])
        r = (focus_point - p1) * 10
        p2 = p1 + r
        res = np.append(res, fan_detector(mesh, p1, p2, width, line_num,  *args, **kwargs), axis=0)
        print('\r', end='')
        print("Generating array of fan detectors: ", str(i*100 // fan_num) + "% complete", end='')
        incline += d_incline
    print('\r \r ', end='')
    print('\r \r ', end='')
    return res


def parallel_detector(mesh, p1, p2, width, number, shift, index=(0, 1), *args, **kwargs):
    """ Creates array of parallel detectors.

       Args:
           mesh(tomomak.main_structures.Mesh): mesh to work with.
           p1(tuple of 2 floats): Detector origin (x, y).
           p2(tuple of 2 floats): Second point, characterizing central axis of detectors.
           width(float): width of each line.
           number(int): number of detectors.
           shift(float): shift of each line as compared to previous.
           index(tuple of two ints, optional): axes to build object at. Default: (0,1).
           *args, **kwarg - additional line2d arguments.

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
        addition = np.array([line2d(mesh, p1, p2, width, index=index,  *args, **kwargs)])
        res = np.append(res, addition, axis=0)
        line = line.parallel_offset(shift, 'left')
    return res


def _line_to_polygon(p1, p2, width, divergence=0):
    """Generate detector geometry for one Line of Sight.

    line of sight can be collimated or diverging (cone-like).

    Args:
        p1, p2(tuple of 2 points):
        width(float): line width.
        divergence(float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
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
