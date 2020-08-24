from . import abstract_axes
import numpy as np
import re
import matplotlib.pyplot as plt
from tomomak.plots import plot1d, plot2d, plot3d
import warnings
from tomomak import util


class Axis1d(abstract_axes.Abstract1dAxis):
    """1D regular or irregular coordinate system.

    Main tomomak coordinate system.
    All transforms from or to other coordinate systems, plotting, etc.
    are performed using this coordinate system as mediator.
    If regular, than it is Cartesian coordinate system.
    """
    def __init__(self, coordinates=None, edges=None, lower_limit=0, upper_limit=None, size=None, name="", units=""):
        super().__init__(name, units)
        if coordinates is not None:
            if size is not None or upper_limit is not None:
                warnings.warn("Since coordinates are given explicitly, size and upper_limit arguments are ignored.")
            if edges is not None:
                warnings.warn("Since coordinates are given explicitly, edges are ignored.")
            self._create_using_coordinates(coordinates, lower_limit)
        elif edges is not None:
            if size is not None or upper_limit is not None:
                warnings.warn("Since coordinates are given explicitly, size and upper_limit arguments are ignored.")
            self._create_using_edges(edges)
        else:
            if size is None:
                warnings.warn("Axis1d init: size was not set. Default size = 10 is used.")
                size = 10
            if upper_limit is None:
                warnings.warn("Axis1d init: upper_limit  was not set. Default upper_limit = 10 is used.")
                upper_limit = 10
            self._create_using_limits(lower_limit, upper_limit, size)

    def _create_using_edges(self, edges):
        coordinates = np.zeros(len(edges) - 1)
        for i, _ in enumerate(coordinates):
            coordinates[i] = (edges[i] + edges[i + 1]) / 2
        self._create_using_coordinates(coordinates, edges[0])
        self._cell_edges[0], self._cell_edges[-1] = edges[0], edges[-1]

    def _create_using_limits(self, lower_limit, upper_limit, size):
        self._size = size
        dv = np.abs(upper_limit - lower_limit) / size
        self._volumes = np.full(size, dv)
        self._coordinates = np.fromfunction(lambda i: lower_limit + (i * dv) + (dv / 2), (size,))
        self._calc_cell_edges(lower_limit)
        self._cell_edges[-1] = upper_limit

    def _create_using_coordinates(self, coordinates, lower_limit):
        if (any(np.diff(coordinates)) < 0 and coordinates[-1] > lower_limit
                or any(np.diff(coordinates)) > 0 and coordinates[-1] < lower_limit):
            raise ValueError("Coordinates are not monotonous.")
        if (coordinates[-1] > lower_limit > coordinates[0]
                or coordinates[-1] < lower_limit < coordinates[0]):
            raise ValueError("lower_limit is inside of the first segment.")
        self._size = len(coordinates)
        self._coordinates = coordinates
        dv = np.diff(coordinates)
        self._volumes = np.zeros(self.size)
        dv0 = coordinates[0] - lower_limit
        self._volumes[0] = 2 * dv0
        for i in range(self.size - 1):
            self._volumes[i + 1] = 2 * dv[i] - self._volumes[i]
        for i, v in enumerate(self._volumes):
            if v <= 0:
                raise ValueError("Point № {} of the coordinates is inside of the previous segment. "
                                 "Increase the distance between the points.".format(i))
        self._calc_cell_edges(lower_limit)

    def _calc_cell_edges(self, lower_limit):
        self._cell_edges = np.zeros(self.size + 1)
        self._cell_edges[0] = lower_limit
        for i in range(self.size):
            self._cell_edges[i + 1] = self._volumes[i] + self._cell_edges[i]
        return self._cell_edges

    def __str__(self):
        if self.regular:
            ax_type = 'regular'
        else:
            ax_type = 'irregular'
        return "{}D {} axis with {} cells. Name: {}. Boundaries: {} {}. " \
            .format(self.dimension, ax_type, self.size, self.name,
                    [self._cell_edges[0], self._cell_edges[-1]], self.units)

    @property
    def volumes(self):
        """See description in abstract axes.
        """
        return self._volumes

    def cartesian_coordinates(self, *axes):
        """See description in abstract axes.
        """
        if not axes:
            return self.coordinates
        for a in axes:
            if type(a) is not Axis1d:
                raise TypeError("Cell edges with such combination of axes are not supported.")
        axes = list(axes)
        axes.insert(0, self)
        coord = [a.coordinates for a in axes]
        return np.array(np.meshgrid(*coord, indexing='ij'))

    @property
    def coordinates(self):
        """See description in abstract axes.
        """
        return self._coordinates

    @property
    def cell_edges(self):
        """See description in abstract axes.
        """
        return self._cell_edges

    @property
    def size(self):
        """See description in abstract axes.
        """
        return self._size

    @property
    def regular(self):
        """See description in abstract axes.
        """
        if all(self._volumes - self._volumes[0] == 0):
            return True
        else:
            return False

    def cell_edges2d(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not type(self):
            raise TypeError("Cell edges with such combination of axes are not supported.")
        shape = (self.size, axis2.size)
        res = np.zeros(shape).tolist()
        edge1 = self.cell_edges
        edge2 = axis2.cell_edges
        for i, row in enumerate(res):
            for j, _ in enumerate(row):
                res[i][j] = [(edge1[i], edge2[j]), (edge1[i + 1], edge2[j]),
                             (edge1[i + 1], edge2[j + 1]), (edge1[i], edge2[j + 1])]
        return res

    def cell_edges3d(self, axis2, axis3):
        """See description in abstract axes.
        """
        if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
            raise TypeError("Cell edges with such combination of axes are not supported.")
        shape = (self.size, axis2.size, axis3.size)
        vertices = np.zeros(shape).tolist()
        faces = np.zeros(shape).tolist()
        edge1 = self.cell_edges
        edge2 = axis2.cell_edges
        edge3 = axis3.cell_edges
        for i, row in enumerate(vertices):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    vertices[i][j][k] = [
                        (edge1[i], edge2[j], edge3[k]), (edge1[i + 1], edge2[j], edge3[k]),
                        (edge1[i + 1], edge2[j + 1], edge3[k]), (edge1[i], edge2[j + 1], edge3[k]),
                        (edge1[i], edge2[j], edge3[k + 1]), (edge1[i + 1], edge2[j], edge3[k + 1]),
                        (edge1[i + 1], edge2[j + 1], edge3[k + 1]), (edge1[i], edge2[j + 1], edge3[k + 1])]
                    faces[i][j][k] = [(0, 3, 2, 1), (4, 5, 6, 7),
                                      (2, 3, 7, 6), (4, 7, 3, 0), (0, 1, 5, 4), (1, 2, 6, 5)]
        return vertices, faces

    def intersection(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not type(self):
            raise TypeError("Cell edges with such combination of axes are not supported.")

        intersection = np.zeros([self.size, axis2.size])

        def inters_len(a_min, a_max, b_min, b_max):
            res = min(a_max, b_max) - max(a_min, b_min)
            if res < 0:
                res = 0
            return res
        j_start = 0
        for i, row in enumerate(intersection):
            for j in range(j_start, len(row)):
                dist = inters_len(self.cell_edges[i], self.cell_edges[i + 1],
                                  axis2.cell_edges[j], axis2.cell_edges[j + 1])
                if not dist and j != j_start:
                    j_start = j-1
                    break
                intersection[i, j] = dist
        return intersection

    def plot1d(self, data, mesh, data_type='solution', filled=True,
               fill_scheme='viridis', edge_color='black', grid=False, equal_norm=False, y_label=None, *args, **kwargs):
        """Create 1D plot of the solution or detector geometry.

        matplotlib bar plot is used. Detector data is plotted on the interactive graph.

        Args:
            data (1D ndarray): data to plot.
            mesh (tomomak mesh): mesh for units extraction.
            data_type (str, optional): type of the data: 'solution' or 'detector_geometry'. Default: solution.
            filled (bool_optional, optional): if true, bars are filled. Color depends on the bar height. Default: True.
            fill_scheme (str, optional): matplotlib fill scheme. Valid only if filled is True. Default: 'viridis'.
            edge_color (str, optional): color of the bar edges. See matplotlib colors for the avaiable options.
                Default: 'black'.
            grid (bool, optional): If true, grid is displayed. Default:False.
            equal_norm (bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry. Default: False.
            y_label (str, optional): y_label caption. Default: automatic.
            *args,**kwargs: additional arguments to pass to matplotlib bar plot.

        Returns:
            matplotlib plot, matplotlib axis
        """
        if data_type == 'solution':
            if y_label is None:
                y_label = r"Density, {}{}".format(self.units, '$^{-1}$')
            plot, ax = plot1d.bar1d(data, self, 'Density', y_label, filled, fill_scheme,
                                    edge_color, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = "Detector 1/{}".format(data.shape[0])
            y_label = util.text.detector_caption(mesh)
            plot, ax, _ = plot1d.detector_bar1d(data, self, title, y_label, filled,
                                                fill_scheme, edge_color, grid, equal_norm, *args, **kwargs)
        else:
            raise ValueError("data type {} is unknown".format(data_type))
        plt.show()
        return plot, ax

    def plot2d(self, axis2, data,  mesh, data_type='solution', style='colormesh',
               fill_scheme='viridis', grid=False, equal_norm=False, title=None, *args, **kwargs):
        """Create 2D plot of the solution or detector geometry.

        matplotlib pcolormesh is used. Detector data is plotted on the interactive graph.

        Args:
            axis2 (tomomak axis): second axis. Only cartesian.Axis1d is supported.
            data (2D ndarray): data to plot.
            mesh (tomomak mesh): mesh to extract additional info.
            data_type (str, optional): type of the data: 'solution' or 'detector_geometry'. Default: solution.
            style (str, optional): Plot style. Available options: 'colormesh', 'contour'. Default: 'colormesh'.
            fill_scheme (str, optional): matplotlib fill scheme. Default: 'viridis'.
            grid (bool, optional): If true, grid is displayed. Default:False.
            equal_norm (bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry. Default: False.
            title (str, optional): solution figure caption. Default: automatic.
            *args,**kwargs: additional arguments to pass to matplotlib pcolormesh.

        Returns:
            matplotlib plot, matplotlib axis
        """
        # if type(axis2) is not Axis1d:
        #     raise NotImplementedError("2D plots with such combination of axes are not supported.")
        if 'cartesian_coordinates' in kwargs:
            raise TypeError('cartesian_coordinates is an invalid keyword argument for plot2d of cartesian axes.')
        if data_type == 'solution':
            if title is None:
                units = util.text.density_units([self.units, axis2.units])
                title = r"Density, {}".format(units)
            plot, ax, fig, cb = plot2d.colormesh2d(data, self, axis2, title, style, fill_scheme, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = 'Detector 1/{}'.format(data.shape[0])
            cb_title = util.text.detector_caption(mesh)
            plot, ax, _ = plot2d.detector_plot2d(data, self, axis2, title, cb_title, style, fill_scheme, grid,
                                                 equal_norm, True, 'colormesh', *args, **kwargs)
        else:
            raise ValueError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax

    def plot3d(self, data, axis2, axis3, mesh, data_type='solution', colormap='blue-red', axes=False,
               interp_size=50, *args, **kwargs):
        """Create 2D plot of the solution or detector geometry.

        Args:
            data (3D ndarray): data to plot.
            axis2 (tomomak axis): second axis. Only cartesian.Axis1d is supported.
            axis3 (tomomak axis): third axis. Only cartesian.Axis1d is supported.
            mesh (tomomak mesh): mesh to extract additional info.
            data_type (str, optional): type of the data: 'solution' or 'detector_geometry'. Default: solution.
            colormap (str, optional): Colormap. Default: 'viridis'.
            axes (bool, optional): If true, axes are shown. Default: False.
            interp_size (int, optional): If at least one of the axes is irregular,
                the new grid wil have  interp_size * interp_size * interp_size dimensions.
            *args: arguments to pass to plot3d.contour3d, detector_contour3d.
            **kwargs: keyword arguments to pass to plot3d.contour3d, detector_contour3d.

        Returns:
            0, 0: placeholder
        """
        # if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
        #     raise NotImplementedError("3D plots with such combination of axes are not supported.")
        x_grid, y_grid, z_grid = self.cartesian_coordinates(axis2, axis3)
        if axes:
            axes = ('{}, {}'.format(self.name, self.units),
                    '{}, {}'.format(axis2.name, axis2.units),
                    '{}, {}'.format(axis3.name, axis3.units))

        if data_type == 'solution':
            # title
            units = util.text.density_units([self.units, axis2.units, axis3.units])
            title = re.sub('[${}]', '', r"   Density, {}".format(units))
            # irregular axes
            if not all((self.regular, axis2.regular, axis3.regular)):
                warnings.warn("Since axes are not regular, linear interpolation with {} points used. "
                              "You can change interpolation size with interp_size attribute.".format(interp_size**3))
                x_grid, y_grid, z_grid, new_data = \
                    util.geometry3d.make_regular(data, x_grid, y_grid, z_grid, interp_size)
            else:
                new_data = data
            # plot
            plot3d.contour3d(new_data, x_grid, y_grid, z_grid,
                             title=title, colormap=colormap, axes=axes, *args, **kwargs)

        elif data_type == 'detector_geometry':
            # title
            title = '   ' + re.sub('[${}]', '', util.text.detector_caption(mesh))
            # irregular ax4s
            if not all((self.regular, axis2.regular, axis3.regular)):
                warnings.warn("Since axes are not regular, linear interpolation with {} points used."
                              "You can change interpolation size with interp_size attribute.".format(interp_size ** 3))
                x_grid_n, y_grid_n, z_grid_n = x_grid, y_grid, z_grid
                new_data = np.zeros((data.shape[0], interp_size,  interp_size,  interp_size))
                # interpolate data for each detector
                print("Start interpolation.")
                for i, d in enumerate(data):
                    x_grid, y_grid, z_grid, new_data[i] \
                        = util.geometry3d.make_regular(d, x_grid_n, y_grid_n, z_grid_n, interp_size)
                    print('\r', end='')
                    print("...", str((i+1) * 100 // data.shape[0]) + "% complete", end='')
                print('\r \r', end='')
            else:
                new_data = data
            plot3d.detector_contour3d(new_data, x_grid, y_grid, z_grid,
                                      title=title, colormap=colormap, axes=axes, *args, **kwargs)
        else:
            raise ValueError('data type {} is unknown'.format(data_type))

        return 0, 0
