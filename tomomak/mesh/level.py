from . import polar
import numpy as np
from . import abstract_axes
from tomomak.plots import plot2d
from tomomak import util
import matplotlib.pyplot as plt
import shapely.geometry.polygon
from scipy import interpolate


class Axis1d(abstract_axes.Abstract1dAxis):
    """ Coordinates for representation of the map of the equal heights (levels).

    2D Implemented combinations:
        level + polar: levels are monotonically increasing or decreasing from the center (highest or lowest point).
        RESOLUTION2D of the polar axis represent number of points for representation of one curved element.
    3D Implemented combinations:
        level + polar + toroidal:
        level + polar system,  rotated toroidally around x=0 point of the level + polar system.
    See abstract_axes.Axis1D for parent attributes description.

        Attributes:
        level_map (2d ndarray): xy map of the levels.
        x (ndarray): 1D x grid at which level_map is represented.
        y (ndarray): 1D y grid at which level_map is represented.
        x_axis (float): x coordinate of the highest/lowest level.
         Required for combination with polar coordinates. Optional.
        y_axis (float): y coordinate of the highest/lowest level.
         Required for combination with polar coordinates. Optional.
        bry_level (float): level at the boundary - only if you want to study space inside of the level.
         Required for combination with polar coordinates. Optional.
        last_level_coordinates (tuple of ndarrays): xy coordinates of the boundary
         - if you want to specify the boundary coordinates. Optional.
         polar_angle_type (str): Approach used for the polar angle calculation (if combination with polar axis is used).
         'eq_angle' - naive method, with equal angles. 'eq_arc'- equal arc length of the contour.
        'eq_vol' equal volume of the cell. Default: 'eq_angle'.
    """

    LEVEL_INTERP_GRID_SIZE = 256
    RESOLUTION2D = 10

    def __init__(self, level_map, x, y, x_axis=None, y_axis=None, last_level_coordinates=None,
                 coordinates=None, edges=None, lower_limit=0, upper_limit=1, size=None, name='', units='',
                 cart_units='a.u.', polar_angle_type='eq_angle'):
        super().__init__(coordinates, edges, lower_limit, upper_limit, size, name, units, True)
        self.cart_units = cart_units
        self.level_map = level_map
        self.x = x
        self.y = y
        self.x_axis = x_axis
        self.y_axis = y_axis
        self._polar_angle_type = polar_angle_type
        if last_level_coordinates is not None:
            self.last_level_coordinates = np.array(last_level_coordinates).T
        else:
            self.last_level_coordinates = None
        self._check_self_consistency()

    def _check_self_consistency(self):
        if self.x_axis > np.amax(self.x) or self.x_axis < np.amin(self.x):
            raise ValueError("X axis coordinate is outside of the x grid.")
        if self.y_axis > np.amax(self.y) or self.y_axis < np.amin(self.y):
            raise ValueError("Y axis coordinate is outside of the y grid.")
        if self._method not in ('eq_angle', 'eq_arc', 'eq_vol'):
            raise ValueError("Incorrect ")

    @property
    def polar_angle_type(self):
        """Returns approach, used for the polar angle calculation.
        """
        return self._polar_angle_type

    def _calc_eq_naive(self, axis2):
        # 2D levels vs rotational symmetry
        if type(axis2) is polar.Axis1d:
            shape = (self.size, axis2.size)
            res = np.zeros(shape).tolist()
            res2d = axis2.RESOLUTION2D

            # Find level contours
            xx, yy = np.meshgrid(self.x, self.y)
            cs = plt.contour(xx, yy, self.level_map, self.cell_edges)
            contours = cs.allsegs

            # Find polar angles
            r_max = np.sqrt((self.x[-1] - self.x[0]) ** 2 + (self.y[-1] - self.y[0]) ** 2) * 2
            original_angles = axis2.cell_edges
            angles = np.zeros(res2d * (len(original_angles) - 1))

            # Add more angles between grid angles in order to correctly represent cell shape
            for i, _ in enumerate(angles):
                orig_ind = int(np.floor(i / res2d))
                angles[i] = original_angles[orig_ind] + i % res2d / res2d * \
                            (original_angles[orig_ind + 1] - original_angles[orig_ind])
            angles = np.append(angles, original_angles[-1])

            # Lines from the center
            section_lines = [shapely.geometry.polygon.LineString([(self.x_axis, self.y_axis),
                                                                  (self.x_axis + r_max * np.cos(ang),
                                                                   self.y_axis + r_max * np.sin(ang))])
                             for ang in angles]
            def find_central_contour(cont):
                """Find contour, closest to the axis.
                """
                d = np.inf
                ind = 0
                for z, c in enumerate(cont):
                    smallest_dist = np.amin(np.abs(c[:, 0] - self.x_axis))
                    if smallest_dist < d:
                        d = smallest_dist
                        ind = z
                return ind

            c2 = None  # second contour (outer)
            for i in range(len(contours) - 1):
                if contours[i]:  # contours not empty - not central element
                    if c2 is not None:
                        c1 = c2  # first contour (inner)
                    else:
                        cent_ind = find_central_contour(contours[i])
                        c1 = shapely.geometry.polygon.LinearRing(contours[i][cent_ind])
                    cent_ind = find_central_contour(contours[i + 1])
                    c2 = shapely.geometry.polygon.LinearRing(contours[i + 1][cent_ind])
                    if i == len(contours) - 2:  # special case for tha outer level
                        if self.last_level_coordinates is not None:
                            c2 = shapely.geometry.polygon.LinearRing(self.last_level_coordinates)
                    inters_points_c1 = [None] * len(section_lines)
                    inters_points_c2 = [None] * len(section_lines)
                    for j, _ in enumerate(section_lines):
                        inters_points_c1[j] = c1.intersection(section_lines[j])
                        inters_points_c2[j] = c2.intersection(section_lines[j])

                    for j in range(len(axis2.cell_edges) - 1):
                        points = []
                        points.append((inters_points_c1[j * res2d].x,
                                       inters_points_c1[j * res2d].y))  # counter-clockwise form the right bottom corner
                        for k in range(res2d + 1):
                            points.append((inters_points_c2[j * res2d + k].x,
                                           inters_points_c2[j * res2d + k].y))
                        for k in range(res2d):
                            points.append((inters_points_c1[(j + 1) * res2d - k].x,
                                           inters_points_c1[(j + 1) * res2d - k].y))
                        res[i][j] = points

                else:  # special case - inner contour is a point (central point)
                    cent_ind = find_central_contour(contours[i + 1])
                    c2 = shapely.geometry.polygon.LinearRing(contours[i + 1][cent_ind])
                    inters_points_c2 = [None] * len(section_lines)
                    for j, _ in enumerate(section_lines):
                        inters_points_c2[j] = c2.intersection(section_lines[j])

                    for j in range(len(axis2.cell_edges) - 1):
                        points = [(self.x_axis, self.y_axis)]
                        for k in range(res2d + 1):
                            points.append((inters_points_c2[j * res2d + k].x,
                                           inters_points_c2[j * res2d + k].y))
                        res[i][j] = points
            plt.close()

            return res
        else:
            raise TypeError("cell_edges2d_cartesian with such combination of axes is not supported.")

    @abstract_axes.precalculated
    def cell_edges2d_cartesian(self, axis2):
        """See description in abstract axes.
        """
        # 2D levels vs rotational symmetry
        if type(axis2) is polar.Axis1d:
            shape = (self.size, axis2.size)
            res = np.zeros(shape).tolist()
            polar_res2d = axis2.RESOLUTION2D
            level_res2d = self.RESOLUTION2D


            # Interpolate level map
            x = np.linspace(self.x[0], self.x[-1], self.LEVEL_INTERP_GRID_SIZE)
            y = np.linspace(self.y[0], self.y[-1], self.LEVEL_INTERP_GRID_SIZE)
            int_spline = interpolate.RectBivariateSpline(self.x, self.y, self.level_map)
            new_level_map = int_spline(x, y)

            # Find contours
            new_levels = util.array_routines.add_inner_points(self.cell_edges, level_res2d)
            contours = util.geometry2d.find_contour_levels(x, y, new_level_map, new_levels, (self.x_axis, self.y_axis),
                                                           point_num=self.LEVEL_INTERP_GRID_SIZE,
                                                           last_level_coordinates=self.last_level_coordinates)

            original_contours = util.geometry2d.find_contour_levels(x, y, new_level_map, self.cell_edges,
                                                                    (self.x_axis, self.y_axis),
                                                                    point_num=self.LEVEL_INTERP_GRID_SIZE,
                                                                    last_level_coordinates=self.last_level_coordinates)

            # Add more angles between grid angles in order to correctly represent cell shape
            original_angles = axis2.cell_edges
            angles = util.array_routines.add_inner_points(original_angles, polar_res2d)

            # Find points, corresponding to  given angles on different contours
            angle_coords = []
            for i, cnt in enumerate(contours):
                    if cnt is not None:
                        contour_len = cnt.length
                        new_coords = []
                        max_ang = angles[-1]
                        for j, ang in enumerate(angles):
                            new_coords.append(cnt.interpolate(ang / max_ang * contour_len))
                        angle_coords.append(new_coords)
                    else:
                        angle_coords.append(None)

            for i in range(len(original_contours) - 1):
                if contours[i] is not None:  # not central element
                    for j in range(len(axis2.cell_edges) - 1):
                        points = []
                        for k in range(level_res2d + 1):  # all but last level points
                            points.append((angle_coords[i * level_res2d + k][j * polar_res2d].x,
                                           angle_coords[i * level_res2d + k][j * polar_res2d].y))
                        for k in range(polar_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_res2d][j * polar_res2d + k].x,
                                           angle_coords[(i + 1) * level_res2d][j * polar_res2d + k].y))
                        for k in range(level_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_res2d - k][(j + 1) * polar_res2d].x,
                                           angle_coords[(i + 1) * level_res2d - k][(j + 1) * polar_res2d].y))
                        for k in range(polar_res2d):
                            points.append((angle_coords[i * level_res2d][(j + 1) * polar_res2d - k].x,
                                           angle_coords[i * level_res2d][(j + 1) * polar_res2d - k].y))
                        res[i][j] = points

                else:  # special case - inner contour is a point (central point)
                    for j in range(len(axis2.cell_edges) - 1):
                        points = [(self.x_axis, self.y_axis)]
                        for k in range(level_res2d):  # all but last level points
                            points.append((angle_coords[i * level_res2d + k + 1][j * polar_res2d].x,
                                           angle_coords[i * level_res2d + k + 1][j * polar_res2d].y))
                        for k in range(polar_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_res2d][j * polar_res2d + k].x,
                                           angle_coords[(i + 1) * level_res2d][j * polar_res2d + k].y))
                        for k in range(level_res2d):
                            points.append((angle_coords[(i + 1) * level_res2d - k][(j + 1) * polar_res2d].x,
                                           angle_coords[(i + 1) * level_res2d - k][(j + 1) * polar_res2d].y))
                        res[i][j] = points

            return res
        else:
            raise TypeError("cell_edges2d_cartesian with such combination of axes is not supported.")

    @abstract_axes.precalculated
    def cell_edges3d_cartesian(self, axis2, axis3):
        """See description in abstract axes.
        """
        raise TypeError("cell_edges3d_cartesian with such combination of axes is not supported.")

    def plot2d(self, axis2, data, mesh, data_type='solution', style='colormesh', fill_scheme='viridis',
               cartesian_coordinates=False, grid=False, equal_norm=False, title=None, *args, **kwargs):
        """See description in abstract axes.
        """
        if cartesian_coordinates:
            if type(axis2) is not polar.Axis1d or not axis2.spatial:
                raise TypeError("2D plots in cartesian coordinates with such combination of axes are not supported.")

            ax_names = ("{}, {}".format('X', self.cart_units), "{}, {}".format('Y', self.cart_units))
            if style == 'colormesh':
                if data_type == 'solution':
                    if title is None:
                        title = util.text.solution_caption(True, self, axis2)
                    plot, ax, _, _ = plot2d.patches(data, self, axis2, title, fill_scheme, None,
                                                    ax_names, *args, **kwargs)
                elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
                    title = 'Detector 1/{}'.format(data.shape[0])
                    cb_title = util.text.detector_caption(mesh, data_type, cartesian=True)
                    plot, ax, _ = plot2d.detector_plot2d(data, self, axis2, title, cb_title, style, fill_scheme,
                                                         grid, equal_norm, False, 'patches', ax_names, *args, **kwargs)
                else:
                    raise ValueError('data type {} is unknown'.format(data_type))
                plt.show()
            else:
                plot, ax = super().plot2d(axis2, data, mesh, data_type, style,
                                          fill_scheme, grid, equal_norm, title, *args, **kwargs)

        else:
            plot, ax = super().plot2d(axis2, data, mesh, data_type, style,
                                      fill_scheme, grid, equal_norm, title, *args, **kwargs)
        return plot, ax

    @abstract_axes.precalculated
    def cartesian_coordinates(self, *axes):
        """See description in abstract axes.
             """
        if not axes:
            raise ValueError("Impossible to convert 1D polar axis to cartesian coordinates.")
        if len(axes) == 1:
            # 2D polar coordinate system
            shape = (self.size, axes[0].size)
            x = np.zeros(shape)
            y = np.zeros(shape)
            if type(axes[0]) is polar.Axis1d:
                cells = self.cell_edges2d_cartesian(axes[0])
                for i, row in enumerate(x):
                    for j, _ in enumerate(row):
                        cell = shapely.geometry.Polygon(cells[i][j])
                        x[i, j] = cell.centroid.x
                        y[i, j] = cell.centroid.y
                return x, y
        raise TypeError("cartesian_coordinate with such combination of axes is not supported.")

    @abstract_axes.precalculated
    def from_cartesian(self, coordinates, *axes):
        """See description in abstract axes.
            """
        if len(axes) == 1:
            x_new, y_new, = coordinates[0], coordinates[1]
            # Level-polar coordinates
            if type(axes[0]) is polar.Axis1d:
                theta = (np.arctan2(y_new - self.y_axis, x_new - self.x_axis) + 2 * np.pi) % (2 * np.pi)
                f = interpolate.interp2d(self.x, self.y, self.level_map, kind='cubic')
                new_levels = np.diagonal(f(x_new, y_new))
                return new_levels, theta
        else:
            raise TypeError("from_cartesian with such combination of axes is not supported.")

