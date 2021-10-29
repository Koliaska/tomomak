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
        if polar_angle_type == 'eq_angle':
            self._resolution2D = 0
        else:
            self._resolution2D = 10
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
        if self._polar_angle_type not in ('eq_angle', 'eq_arc', 'eq_vol'):
            raise ValueError("Incorrect ")

    @property
    def polar_angle_type(self):
        """Returns approach, used for the polar angle calculation.
        """
        return self._polar_angle_type

    @property
    def resolution2d(self):
        """Number of points for curved lines representation.
        If polar_angle_type == 'eq_angle', it is 0, otherwise it can be changed.

        Returns:
        Resolution2d(int): number of point between two extreme level points of the cell.
        """
        return self._resolution2D

    @resolution2d.setter
    def resolution2d(self, value):
        if self.polar_angle_type == 'eq_angle':
            raise ValueError("resolution2D cannot be changed for such polar_angle_type method.")
        self._resolution2D = value

    @abstract_axes.precalculated
    def cell_edges2d_cartesian(self, axis2):
        """See description in abstract axes.
        """
        # 2D levels vs rotational symmetry
        if type(axis2) is polar.Axis1d:

            shape = (self.size, axis2.size)
            res = np.zeros(shape).tolist()
            polar_res2d = axis2.RESOLUTION2D - 1
            polar_step = polar_res2d + 1

            # Interpolate level map
            x = np.linspace(self.x[0], self.x[-1], self.LEVEL_INTERP_GRID_SIZE)
            y = np.linspace(self.y[0], self.y[-1], self.LEVEL_INTERP_GRID_SIZE)
            int_spline = interpolate.RectBivariateSpline(self.x, self.y, self.level_map)
            new_level_map = int_spline(x, y)

            # Add more angles between grid angles in order to correctly represent cell shape
            original_angles = axis2.cell_edges
            angles = util.array_routines.add_inner_points(original_angles, polar_res2d)

            if self.polar_angle_type == 'eq_arc' or self.polar_angle_type == 'eq_vol':
                level_res2d = self.resolution2d
                point_num = self.LEVEL_INTERP_GRID_SIZE
                interp_angles = None
            else:  # self.polar_angle_type == 'eq_angle'
                level_res2d = 0
                point_num = len(angles)
                interp_angles = angles

            level_step = level_res2d + 1
            new_levels = util.array_routines.add_inner_points(self.cell_edges, level_res2d)

            # Find contours
            contours = util.geometry2d.find_contour_levels(x, y, new_level_map, new_levels, (self.x_axis, self.y_axis),
                                                           point_num=point_num,
                                                           last_level_coordinates=self.last_level_coordinates,
                                                           angles=interp_angles)

            original_contours = util.geometry2d.find_contour_levels(x, y, new_level_map, self.cell_edges,
                                                                    (self.x_axis, self.y_axis),
                                                                    point_num=point_num,
                                                                    last_level_coordinates=self.last_level_coordinates,
                                                                    angles=interp_angles)

            # Find points, corresponding to  given angles on different contours
            angle_coords = []
            for i, cnt in enumerate(contours):
                if cnt is not None:
                    new_coords = []
                    if self.polar_angle_type == 'eq_arc' or self.polar_angle_type == 'eq_vol':
                        contour_len = cnt.length
                        for j, ang in enumerate(angles):
                            new_coords.append(cnt.interpolate(ang / (2 * np.pi) * contour_len))
                    else:  # self.polar_angle_type == 'eq_angle'
                        for j, ang in enumerate(angles):
                            new_coords.append(shapely.geometry.Point(cnt.coords[j]))
                    angle_coords.append(new_coords)
                else:
                    angle_coords.append(None)

            # Start procedure
            for i in range(len(original_contours) - 1):
                if contours[i] is not None:  # not central element
                    for j in range(len(axis2.cell_edges) - 1):
                        points = []
                        for k in range(level_res2d + 1):
                            points.append((angle_coords[i * level_step + k][j * polar_step].x,
                                           angle_coords[i * level_step + k][j * polar_step].y))
                        for k in range(polar_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_step][j * polar_step + k].x,
                                           angle_coords[(i + 1) * level_step][j * polar_step + k].y))
                        for k in range(level_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_step - k][(j + 1) * polar_step].x,
                                           angle_coords[(i + 1) * level_step - k][(j + 1) * polar_step].y))
                        for k in range(polar_res2d + 1):
                            points.append((angle_coords[i * level_step][(j + 1) * polar_step - k].x,
                                           angle_coords[i * level_step][(j + 1) * polar_step- k].y))
                        res[i][j] = points

                else:  # special case - inner contour is a point (central point)
                    for j in range(len(axis2.cell_edges) - 1):
                        points = [(self.x_axis, self.y_axis)]
                        for k in range(level_res2d):
                            points.append((angle_coords[i * level_step + k + 1][j * polar_step].x,
                                           angle_coords[i * level_step + k + 1][j * polar_step].y))
                        for k in range(polar_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_step][j * polar_step + k].x,
                                           angle_coords[(i + 1) * level_step][j * polar_step + k].y))
                        for k in range(level_res2d + 1):
                            points.append((angle_coords[(i + 1) * level_step - k][(j + 1) * polar_step].x,
                                           angle_coords[(i + 1) * level_step - k][(j + 1) * polar_step].y))
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


