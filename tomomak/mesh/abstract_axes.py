from abc import ABC, abstractmethod
import numpy as np
import re
from functools import wraps
import matplotlib.pyplot as plt
from tomomak.plots import plot1d, plot2d, plot3d
import warnings
from tomomak import util
from tomomak.mesh import cartesian


class AbstractAxis(ABC):
    """Superclass for every axis.
    One axis always corresponds to one data array dimension, however may corresponds to several real-world dimensions.
    If you want to create new coordinate system, inherit Abstract1DAxis, Abstract2dAxis or Abstract3dAxis.

    """

    def __init__(self, name="", units="", spatial=True):
        self.name = name
        self.units = units
        self._spatial = spatial

    @property
    @abstractmethod
    def dimension(self):
        """Get number of dimensions.

        Returns:
            int: number of dimensions
        """

    @property
    @abstractmethod
    def spatial(self):
        """Check, if this is a spatial axis.

        Spatial axes may be used for the spatial geometry calculations.

        Returns:
            bool: True if spatial, False, if not.
        """

    @property
    @abstractmethod
    def cell_edges(self):
        """Returns edges of each cell in self coordinates.

        Returns:
            1D iterable of cell edges.

        """

    @property
    @abstractmethod
    def volumes(self):
        """Get volumes (lengths/areas/volumes) of each cell.

        Returns:
            ndarray of floats: cell volumes.
        """

    @property
    @abstractmethod
    def total_volume(self):
        """Get total length/area/volume of axis.

            Returns:
                float: total volume.
        """

    @abstractmethod
    def cartesian_coordinates(self, *axes):
        """Get cartesian coordinates of each cell.

        Args:
            *axes (tomomak axis): additional tomomak axes.

        Returns:
           ndarray: cartesian coordinates of each cell centers in the form of numpy.meshgrid.
        """

    @abstractmethod
    def from_cartesian(self, coordinates, *axes):
        """Converts input array to self coordinates from cartesian coordinates.

        Used only for correct 3D interpolation in order to exclude points outside of the grid.
            (so only conversion in 3D case is needed at the moment)

        Args:
            coordinates (list of meshgrids): list of each coordinate in the form of numpy.meshgrid.
            e.g. for 3D coordinates [xx, yy, zz]
            *axes: (tomomak axis): additional tomomak axes.

        Returns:
            ndarray: self coordinates of each cell centers in the form of numpy.meshgrid.
        """

    @property
    @abstractmethod
    def coordinates(self):
        """Get cartesian coordinates of each cell.

        Returns:
           iterable of floats or points (list of floats): coordinates of each cell centers.
        """

    @property
    @abstractmethod
    def size(self):
        """Get number of axis cells.

        Returns:
            int: number of axis cells.Data array, corresponding to this axis should have same size.
        """

    @property
    @abstractmethod
    def regular(self):
        """If the axis is regular?

        Useful for different transformations.

        Returns:
            boolean: True if the axis is regular. False otherwise.
        """

    @abstractmethod
    def intersection(self, axis2):
        """Intersection length/area/volume of each cell with each cell of another axis of a same type as 2D array.

        Args:
            axis2 (tomomak axis): another axis of a same type.

        Returns:
            2D ndarray: intersection length/area/volume of element i with element j.

        """


def precalculated(method):
    @wraps(method)
    def _impl(self, *args, **kwargs):
        def precalc():
            setattr(self, stored_name, method(self, *args, **kwargs))
            setattr(self, stored_name + '_args', args)
            setattr(self, stored_name + '_kwargs', kwargs)
            return getattr(self, stored_name)

        name = method.__name__
        stored_name = '_' + name
        try:
            stored_res = getattr(self, stored_name)
        except AttributeError:
            return precalc()
        parameters_match = True
        try:
            arg_list = getattr(self, stored_name + '_args')
            for a1, a2 in zip(arg_list, args):
                if type(a1) is list or tuple:
                    if not np.all(np.array(a1) == np.array(a2)):
                        parameters_match = False
                else:
                    if not np.all(a1 == a2):
                        parameters_match = False
        except AttributeError:
            pass
        try:
            kw_list = getattr(self, stored_name + '_kwargs')
            if kw_list != kwargs:
                parameters_match = False
        except AttributeError:
            pass
        if stored_res is None or not parameters_match:
            return precalc()
        return stored_res

    return _impl


def mult_out(func):
    """ Decorator, indicating that function returns multiple output.
    Needed in methods such as axes_method3d or axes_method2d in order to correctly iterate over axes.
    """

    def multiple_output(*args, **kwargs):
        return func(*args, **kwargs)

    return multiple_output


class Abstract1dAxis(AbstractAxis):
    """Superclass for 1D axis.

    1D means that one data array dimension (solution or detector geometry) is describing one real-world dimension.
    Axes need to be stacked, e.g. six 1D cartesian axes describe 6D space in a real world.
    """

    def __init__(self, coordinates=None, edges=None, lower_limit=0, upper_limit=None, size=None, name="", units="",
                 spatial=True, cart_units='a.u.'):
        self.cart_units = cart_units
        super().__init__(name, units, spatial)
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
        self._size = len(edges) - 1
        edges = np.array(edges)
        self._cell_edges = edges
        coordinates = np.zeros(self._size)
        for i, _ in enumerate(coordinates):
            coordinates[i] = (edges[i] + edges[i + 1]) / 2
        self._coordinates = coordinates
        self._volumes = edges[1:] - edges[0:-1]
        # self._create_using_coordinates(coordinates, edges[0])
        # self._cell_edges[0], self._cell_edges[-1] = edges[0], edges[-1]

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

    @property
    def total_volume(self):
        """See description in abstract axes.
        """
        v = self.cell_edges
        v = np.abs(v[-1] - v[0])
        return v

    @property
    def spatial(self):
        """See description in abstract axes.
        """
        return self._spatial

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
    def dimension(self):
        return 1

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

    def intersection(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not type(self):
            raise TypeError("Cell edges with such combination of axes is not supported.")

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
                    j_start = j - 1
                    break
                intersection[i, j] = dist
        return intersection

    def find_position(self, val):
        """Find position of the value in respect to cell edges

        Args:
            val (float): value of interest.

        Returns:
            (int, int): index of the left and right cell edge.
             If value is located exactly at at the cell edge, both values are equal.
             If value is outside of axis boundaries, return (-1, 0) or (0, -1).
        """
        edges = np.array(self.cell_edges)
        if val in edges:
            index = np.searchsorted(edges, val)
            return index, index
        else:
            edges -= val
            if edges[0] > 0:
                return -1, 0
            if edges[-1] < 0:
                return 0, -1
            index = 0
            for i, e in enumerate(edges):
                if e > 0:
                    index = i
                    break
            return index - 1, index

    def plot1d(self, data, mesh, data_type='solution', filled=True,
               fill_scheme='viridis', edge_color='black', grid=False, equal_norm=False, y_label=None, *args, **kwargs):
        """Create 1D plot of the solution or detector geometry.

        matplotlib bar plot is used. Detector data is plotted on the interactive graph.

        Args:
            data (1D ndarray): data to plot.
            mesh (tomomak mesh): mesh for units extraction.
            data_type (str, optional): type of the data: 'solution', 'detector_geometry' or 'detector_geometry_n'.
                Default: solution.
            filled (bool_optional, optional): if true, bars are filled. Color depends on the bar height. Default: True.
            fill_scheme (str, optional): matplotlib fill scheme. Valid only if filled is True. Default: 'viridis'.
            edge_color (str, optional): color of the bar edges. See matplotlib colors for the avaiable options.
                Default: 'black'.
            grid (bool, optional): If true, grid is displayed. Default:False.
            equal_norm (bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry or detector_geometry_n. Default: False.
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
        elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
            title = "Detector 1/{}".format(data.shape[0])
            y_label = util.text.detector_caption(mesh, data_type)
            plot, ax, _ = plot1d.detector_bar1d(data, self, title, y_label, filled,
                                                fill_scheme, edge_color, grid, equal_norm, *args, **kwargs)
        else:
            raise ValueError("data type {} is unknown".format(data_type))
        plt.show()
        return plot, ax

    def plot2d(self, axis2, data, mesh, data_type='solution', style='colormesh',
               fill_scheme='viridis', grid=False, equal_norm=False, title=None, ax_names=None, *args, **kwargs):
        """Create 2D plot of the solution or detector geometry.

        matplotlib pcolormesh is used. Detector data is plotted on the interactive graph.

        Args:
            axis2 (tomomak axis): second axis.
            data (2D ndarray): data to plot.
            mesh (tomomak mesh): mesh to extract additional info.
            data_type (str, optional): type of the data: 'solution', 'detector_geometry' or 'detector_geometry_n'.
                Default: solution.
            style (str, optional): Plot style. Available options: 'colormesh', 'contour'. Default: 'colormesh'.
            fill_scheme (str, optional): matplotlib fill scheme. Default: 'viridis'.
            grid (bool, optional): If true, grid is displayed. Default:False.
            equal_norm (bool, optional): If true, all detectors will have same norm.
                Valid only if data_type = detector_geometry or detector_geometry_n. Default: False.
            title (str, optional): solution figure caption. Default: automatic.
            ax_names (list of str, optional): caption for coordinate axes. Default: automatic.
            *args,**kwargs: additional arguments to pass to matplotlib pcolormesh.

        Returns:
            matplotlib plot, matplotlib axis
        """
        # if type(axis2) is not Axis1d:
        #     raise NotImplementedError("2D plots with such combination of axes are not supported.")
        if 'cartesian_coordinates' in kwargs:
            raise TypeError('cartesian_coordinates is an invalid keyword argument for plot2d of cartesian axes.')
        if ax_names is None:
            ax_names = ("{}, {}".format(self.name, self.units), "{}, {}".format(axis2.name, axis2.units))
        if data_type == 'solution':
            if title is None:
                title = util.text.solution_caption(False, self, axis2)
            plot, ax, fig, cb = plot2d.colormesh2d(data, self, axis2, title, style, fill_scheme, grid, None, ax_names,
                                                   *args, **kwargs)
        elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
            title = 'Detector 1/{}'.format(data.shape[0])
            cb_title = util.text.detector_caption(mesh, data_type)
            plot, ax, _ = plot2d.detector_plot2d(data, self, axis2, title, cb_title, style, fill_scheme, grid,
                                                 equal_norm, True, 'colormesh', ax_names, *args, **kwargs)
        else:
            raise ValueError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax

    def plot3d(self, data, axis2, axis3, mesh, data_type='solution', colormap='blue-red', axes=False,
               cartesian_coordinates=False, interp_size=None, ax_names=None, style=0, *args, **kwargs):
        """Create 3D plot of the solution or detector geometry.

        Args:
            data (3D ndarray): data to plot.
            axis2 (tomomak axis): second axis.
            axis3 (tomomak axis): third axis.
            mesh (tomomak mesh): mesh to extract additional info.
            data_type (str, optional): type of the data: 'solution', 'detector_geometry' or 'detector_geometry_n'.
                Default: solution.
            colormap (str, optional): Colormap. Default: 'viridis'.
            axes (bool, optional): If true, axes are shown. Default: False.
            cartesian_coordinates (bool, optional): If true, data is converted to cartesian coordinates. Default: False.
            interp_size (int, optional): If at least one of the axes is irregular,
                the new grid wil have  interp_size * interp_size * interp_size dimensions.
            ax_names (list of str, optional): caption for coordinate axes. Default: automatic naming.
            style (int): plot style. See plotting 3D example.
            *args: arguments to pass to plot3d.contour3d, detector_contour3d.
            **kwargs: keyword arguments to pass to plot3d.contour3d, detector_contour3d.

        Returns:
            0, 0: placeholder
        """
        # if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
        #     raise NotImplementedError("3D plots with such combination of axes are not supported.")
        # x_grid, y_grid, z_grid = self.cartesian_coordinates(axis2, axis3)
        # Title
        if data_type == 'solution':
            title = util.text.solution_caption(cartesian_coordinates, self, axis2, axis3).replace('$', '') \
                .replace('{', '').replace('}', '')
        elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
            title = '   ' + re.sub('[${}]', '', util.text.detector_caption(mesh, data_type, cartesian_coordinates))
        if axes:
            if ax_names is None:
                axes = ('{}, {}'.format(self.name, self.units),
                        '{}, {}'.format(axis2.name, axis2.units),
                        '{}, {}'.format(axis3.name, axis3.units))
            else:
                axes = ax_names
        # Voxel style
        if style == 0:
            if cartesian_coordinates:
                vertices, faces = self.cell_edges3d_cartesian(axis2, axis3)
            else:
                vertices, faces = self.cell_edges3d(axis2, axis3)
            vertices = np.array(vertices)
            if data_type == 'solution':
                x = []
                y = []
                z = []
                new_data = []
                new_faces = []
                shift = 0
                for i, a1 in enumerate(vertices):
                    for j, a2 in enumerate(a1):
                        for k, a3 in enumerate(a2):
                            vert_faces = np.array(faces[i][j][k]) + shift
                            for f in vert_faces:
                                new_faces.append(f)
                            for vert in a3:
                                x.append(vert[0])
                                y.append(vert[1])
                                z.append(vert[2])
                                new_data.append(data[i][j][k])
                                shift += 1
                plot3d.voxel_plot(new_data, x, y, z, new_faces, title=title, axes=axes, colormap=colormap,
                                  *args, **kwargs)
            elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
                new_data = []
                for det in data:
                    x = []
                    y = []
                    z = []
                    det_data = []
                    new_faces = []
                    shift = 0
                    for i, a1 in enumerate(vertices):
                        for j, a2 in enumerate(a1):
                            for k, a3 in enumerate(a2):
                                vert_faces = np.array(faces[i][j][k]) + shift
                                for f in vert_faces:
                                    new_faces.append(f)
                                for vert in a3:
                                    x.append(vert[0])
                                    y.append(vert[1])
                                    z.append(vert[2])
                                    det_data.append(det[i][j][k])
                                    shift += 1
                    new_data.append(det_data)
                plot3d.detector_voxel_plot(new_data, x, y, z, new_faces, title=title, axes=axes, colormap=colormap,
                                           *args, **kwargs)
            else:
                raise ValueError('data type {} is unknown'.format(data_type))
            return 0, 0

        if cartesian_coordinates:
            x_grid, y_grid, z_grid = self.cartesian_coordinates(axis2, axis3)
        else:
            coord = [self.coordinates, axis2.coordinates, axis3.coordinates]
            x_grid, y_grid, z_grid = np.array(np.meshgrid(*coord, indexing='ij'))

        if data_type == 'solution':
            # irregular or non-cartesian axes
            if not all((self.regular, axis2.regular, axis3.regular)) or \
                    (cartesian_coordinates and not all(type(x) == cartesian.Axis1d for x in (self, axis2, axis3))):
                if interp_size is None:
                    interp_size = 50
                    warnings.warn("Since axes are not regular, linear interpolation with {} points used. "
                                  "You can change interpolation size with interp_size attribute."
                                  .format(interp_size ** 3))
                x_grid, y_grid, z_grid, new_data = \
                    util.geometry3d_basic.make_regular(data, x_grid, y_grid, z_grid, interp_size)
                new_data = np.nan_to_num(new_data)
                new_data = np.clip(new_data, np.amin(data), np.amax(data))
                mask = mesh.is_in_grid(self.from_cartesian([x_grid, y_grid, z_grid], axis2, axis3), self, axis2, axis3)
                new_data *= mask
            else:
                new_data = data
            # plot
            plot3d.contour3d(new_data, x_grid, y_grid, z_grid,
                             title=title, colormap=colormap, axes=axes, style=style, *args, **kwargs)

        elif data_type == 'detector_geometry' or data_type == 'detector_geometry_n':
            # irregular axes
            if not all((self.regular, axis2.regular, axis3.regular)) or \
                    (cartesian_coordinates and not all(type(x) == cartesian.Axis1d for x in (self, axis2, axis3))):
                if interp_size is None:
                    interp_size = 50
                    warnings.warn("Since axes are not regular, linear interpolation with {} points used. "
                                  "You can change interpolation size with interp_size attribute."
                                  .format(interp_size ** 3))
                x_grid_n, y_grid_n, z_grid_n = x_grid, y_grid, z_grid
                new_data = np.zeros((data.shape[0], interp_size, interp_size, interp_size))
                # interpolate data for each detector
                print("Start interpolation.")
                mask = None
                for i, d in enumerate(data):
                    x_grid, y_grid, z_grid, new_data[i] \
                        = util.geometry3d_basic.make_regular(d, x_grid_n, y_grid_n, z_grid_n, interp_size)
                    if mask is None:
                        mask = mesh.is_in_grid(self.from_cartesian([x_grid, y_grid, z_grid], axis2, axis3), self, axis2,
                                               axis3)
                    new_data[i] = np.nan_to_num(new_data[i])
                    new_data[i] = np.clip(new_data[i], np.amin(data[i]), np.amax(data[i]))
                    new_data[i] *= mask
                    print('\r', end='')
                    print("...", str((i + 1) * 100 // data.shape[0]) + "% complete", end='')
                print('\r \r', end='')

            else:
                new_data = data
            plot3d.detector_contour3d(new_data, x_grid, y_grid, z_grid,
                                      title=title, colormap=colormap, axes=axes, style=style, *args, **kwargs)
        else:
            raise ValueError('data type {} is unknown'.format(data_type))

        return 0, 0

    @abstractmethod
    def cell_edges2d_cartesian(self, axis2):
        """Get edges of a cell on a mesh consisting of this and another 1D axis in cartesian coordinates.

        Args:
            axis2 (tomomak axis): another 1D axis of the same or other type.

        Returns:
            2D list of lists of points (x, y) in cartesian coordinates: points of the polygon, representing the cell.
            Size of the list is self.size x axis1.size
        """

    @abstractmethod
    def cell_edges3d_cartesian(self, axis2, axis3):
        """Get edges of a cell on a mesh consisting of this and two other 1D axis in cartesian coordinates.

        Args:
            axis2 (tomomak axis): another 1D axes of the same or other type.
            axis3 (tomomak axis): another 1D axes of the same or other type.

        Returns:
            (vertices, faces)
            vertices: 3D list of lists of points (x, y, z) in cartesian coordinates: vertices the cell.
            Size of the list is self.size x axis1.size x axis3.size
            faces: 3D list of lists of cell faces. Each face is a list of  3 vertices, denoted in cw direction.
            Some methods may work with faces=None
             - in this case faces are created automatically for the convex hull of the vertices.
            Size of the list is self.size x axis1.size x axis3.size

        """

    def cell_edges3d(self, axis2, axis3):
        """Get edges of a cell on a mesh consisting of this and two other 1D axis in self coordinates.

        Args:
            axis2 (tomomak axis): another 1D axes of the same or other type.
            axis3 (tomomak axis): another 1D axes of the same or other type.

        Returns:
            (vertices, faces)
            vertices: 3D list of lists of points (x, y, z) in cartesian coordinates: vertices the cell.
            Size of the list is self.size x axis1.size x axis3.size
            faces: 3D list of lists of cell faces. Each face is a list of vertices, denoted in cw direction.
            Some methods may work with faces=None
             - in this case faces are created automatically for the convex hull of the vertices.
            Size of the list is self.size x axis1.size x axis3.size

        """
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
                    faces[i][j][k] = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
                                      (2, 3, 7), (2, 7, 6), (4, 7, 3), (4, 3, 0),
                                      (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5)]
        return vertices, faces


class Abstract2dAxis(AbstractAxis):
    @property
    def dimension(self):
        return 2

    @abstractmethod
    def plot2d(self, data, *args, **kwargs):
        """

        :return:
        """

    @abstractmethod
    def plot3d(self, data, axis2, *args, **kwargs):
        """

        :return:
        """

    @abstractmethod
    def cell_edges2d_cartesian(self):
        """

        Args:

        Returns:

        """

    @abstractmethod
    def cell_edges3d_cartesian(self, axis2):
        """

        Args:
            axis2:

        Returns:

        """


class Abstract3dAxis(AbstractAxis):
    @property
    def dimension(self):
        return 3

    @abstractmethod
    def plot3d(self, data, *args, **kwargs):
        """

        :return:
        """

    @abstractmethod
    def cell_edges3d_cartesian(self):
        """

        Args:

        Returns:

        """
