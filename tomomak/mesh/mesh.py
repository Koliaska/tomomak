import numpy as np
from tomomak.util import array_routines
import itertools


class Mesh:
    """Mesh, containing all coordinate axes.

        mesh is a top-level container for all coordinate axes. It is one of the core TOMOMAK structures.
        Allows integration and summation over needed axes and prepares solution and detector_geometry data for plotting.
        Usually used as part of the Model in order to allow binding to the real-world coordinates
        and hence visualisation, generation of detector geometry or test and apriori objects.
        mesh doesn't have public attributes. Access to data is provided via properties.
        One axis always corresponds to one dimension in solution and detector_geometry arrays,
        however it may correspond to several dimensions in the real world coordinates.
        """

    def __init__(self, axes=()):
        """Constructor requires only list of coordinate axes. Later axes may be added or removed.

        Args:
            axes(tuple of tomomak axes, optional): Tuple of 1D, 2D or 3D axes.
                The order of the axes will be preserved. default: ().
        """
        self._axes = []
        self._dimension = 0
        for axis in axes:
            self.add_axis(axis)

    def __str__(self):
        res = "{}D mesh with {} axes:\n".format(self.dimension, len(self.axes))
        for i, ax in enumerate(self.axes):
            res += "{}. {} \n".format(i+1, ax)
        return res

    @property
    def dimension(self):
        """int: Number of mesh dimensions.

        Note that number of axes may be smaller than number of dimensions,
        since axis is allowed to represent more than one dimension.
        """
        return self._dimension

    @property
    def axes(self):
        """list of tomomak axes: mesh axes.
        """
        return self._axes

    @property
    def shape(self):
        """tuple of ints: tuple of each axis dimensions.

        In the Model this parameter is also equal to the shape of solution.
        """
        return tuple([ax.size for ax in self.axes])

    def add_axis(self, axis, index=None):
        if index is None:
            index = len(self._axes)
        self._axes.insert(index, axis)
        self._dimension += axis.dimension

    def remove_axis(self, index=-1):
        self._dimension -= self._axes[index].dimension
        del self._axes[index]

    def integrate(self, data, index, integrate_type='integrate'):
        """ Calculates sum of data * dv or sum of data over given axes,
        where dv is length/surface/volume of a cell.

        Args:

            data (numpy.array): Array to integrate.
            index (int/list of int): Index or list of indexes of ndarray dimension to integrate over.
            integrate_type ('integrate', 'sum'): Type of integration.
                'integrate': calculates sum of data * dv. Used for solution integration.
                'sum': returns sum over one dimension. Used for detector_geometry integration.

        Returns:
            numpy.array: Integrated or summed array.
        Raises:
            AttributeError: if integration type is unknown.
        """

        if integrate_type not in ['integrate', 'sum']:
            raise AttributeError('Integration type is unknown.')
        if isinstance(index, int):
            index = [index]
        axis_shift = 0
        for i in index:
            axis = i + axis_shift
            if integrate_type == 'integrate':
                dv = self.axes[i].volumes
                data = array_routines.multiply_along_axis(data, dv, axis)
            data = np.sum(data, axis)
            axis_shift -= 1
        return data

    def _other(self, index):
        if isinstance(index, int):
            index = [index]
        invert_index = []
        for axis_index, _ in enumerate(self.axes):
            if all(axis_index != i for i in index):
                invert_index.append(axis_index)
        return invert_index

    def integrate_other(self, data, index):
        invert_index = self._other(index)
        return self.integrate(data, invert_index)

    def sum_other(self, data, index):
        invert_index = self._other(index)
        return self.integrate(data, invert_index, integrate_type='sum')

    def _prepare_data(self, data, index, data_type):
        if data_type == 'solution':
            new_data = self.integrate_other(data, index)
        elif data_type == 'detector_geometry':
            shape = [data.shape[0]]
            for i in index:
                shape.append(data.shape[i + 1])
            new_data = np.zeros(shape)
            for i, d in enumerate(data):
                new_data[i] = self.sum_other(d, index)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        return new_data

    def plot1d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        new_data = self._prepare_data(data, index, data_type)
        plot = self._axes[index[0]].plot1d(new_data, data_type, *args, **kwargs)
        return plot

    def plot2d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if len(index) == 1:
            try:
                new_data = self._prepare_data(data, index[0], data_type)
                plot = self._axes[index[0]].plot2d(new_data, *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError):
                index.append(index[0] + 1)
        # try to draw using 2 axes
        new_data = self._prepare_data(data, index, data_type)
        try:
            plot = self._axes[index[0]].plot2d(new_data, self._axes[index[1]], data_type, *args, **kwargs)
        except NotImplementedError:
            new_data = new_data.transpose()
            plot = self._axes[index[1]].plot2d(new_data, self._axes[index[0]], data_type, *args, **kwargs)
        return plot

    def plot3d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if len(index) == 1:
            try:
                new_data = self._prepare_data(data, index[0], data_type)
                plot = self._axes[index[0]].plot3d(new_data, *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError):
                index.append(index[0] + 1)
        # try to draw using 2 axes
        new_data = self._prepare_data(data, index, data_type)
        try:
            plot = self._axes[index[0]].plot3d(new_data, self._axes[index[1]], data_type, *args, **kwargs)
            return plot
        except (NotImplementedError,  TypeError):
            try:
                new_data = new_data.transpose()
                plot = self._axes[index[1]].plot3d(new_data, self._axes[index[0]], data_type, *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError):
                index.append(index[1] + 1)
        # try to draw using 3 axes
        new_data = self._prepare_data(data, index, data_type)
        ind_lst = list(itertools.permutations((0, 1, 2), 3))
        for p in ind_lst:
            try:
                new_ax = [self._axes[i] for i in p]
                dat = np.array(new_data)
                for i in range(3):
                    np.moveaxis(dat, p[i], i)
                plot = new_ax[index[0]].plot3d(dat, new_ax[index[1]], new_ax[index[2]], data_type, *args, **kwargs)
                return plot
            except NotImplementedError:
                pass
        raise TypeError("plot3d is not implemented for such axes combination or other problem occurred.")

    def draw_mesh(self):
        pass

    def density(self, data, coordinate):
        pass
