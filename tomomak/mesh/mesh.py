import numpy as np
from tomomak.util import array_routines
import itertools
from collections.abc import Iterable


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
            axes (tuple of tomomak axes, optional): Tuple of 1D, 2D or 3D axes.
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
            ValueError: if integration type is unknown.
        """

        if integrate_type not in ['integrate', 'sum']:
            raise ValueError('Integration type is unknown.')
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

    @staticmethod
    def _move_unordered_axes(ar, index):
        new_index = np.array(index)
        new_index_s = np.sort(new_index)
        delta = np.zeros_like(new_index)
        for i, elem in enumerate(new_index):
            loc = np.where(new_index_s == elem)[0]
            delta[i] = elem - loc
        new_index -= delta
        return np.moveaxis(ar, range(len(new_index)), new_index)

    def integrate_other(self, data, index):
        invert_index = self._other(index)
        ar = self.integrate(data, invert_index)
        return self._move_unordered_axes(ar, index)

    def sum_other(self, data, index):
        invert_index = self._other(index)
        ar = self.integrate(data, invert_index, integrate_type='sum')
        return self._move_unordered_axes(ar, index)

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
            raise ValueError('data type {} is unknown'.format(data_type))
        return new_data

    def plot1d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        new_data = self._prepare_data(data, index, data_type)
        plot = self._axes[index[0]].plot1d(new_data, self, data_type, *args, **kwargs)
        return plot

    def plot2d(self, data, index=(0, 1), data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if len(index) == 1:
            try:
                new_data = self._prepare_data(data, index[0], data_type)
                plot = self._axes[index[0]].plot2d(new_data, self, data_type,  *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError, AttributeError):
                index.append(index[0] + 1)
        # try to draw using 2 axes
        new_data = self._prepare_data(data, index, data_type)
        try:
            plot = self._axes[index[0]].plot2d(self._axes[index[1]], new_data, self, data_type, *args, **kwargs)
        except (NotImplementedError, TypeError, AttributeError):
            if data_type == 'solution':
                new_data = new_data.transpose()
            else:
                new_data = new_data.transpose((0, 2, 1))
            plot = self._axes[index[1]].plot2d(self._axes[index[0]], new_data, self, data_type, *args, **kwargs)
        return plot

    def plot3d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if len(index) == 1:
            try:
                new_data = self._prepare_data(data, index[0], data_type)
                plot = self._axes[index[0]].plot3d(new_data, self, *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError):
                index.append(index[0] + 1)
        # try to draw using 2 axes
        new_data = self._prepare_data(data, index, data_type)
        try:
            plot = self._axes[index[0]].plot3d(new_data, self._axes[index[1]], self, data_type, *args, **kwargs)
            return plot
        except (NotImplementedError,  TypeError, AttributeError):
            try:
                new_data = new_data.transpose()
                plot = self._axes[index[1]].plot3d(new_data, self._axes[index[0]], self, data_type, *args, **kwargs)
                return plot
            except (NotImplementedError, TypeError, AttributeError):
                index.append(index[1] + 1)
        # try to draw using 3 axes
        if len(self.axes) > 2:
            new_data = self._prepare_data(data, index, data_type)
            ind_lst = list(itertools.permutations((0, 1, 2), 3))
            for p in ind_lst:
                try:
                    new_ax = [self._axes[i] for i in p]
                    dat = np.array(new_data)
                    for i in range(3):
                        dat = np.moveaxis(dat, p[i], i)
                    plot = new_ax[index[0]].plot3d(dat, new_ax[index[1]], new_ax[index[2]],
                                                   self, data_type, *args, **kwargs)
                    return plot
                except (NotImplementedError, TypeError):
                    pass
        raise TypeError("plot3d is not implemented for such axes combination or other problem occurred.")

    def axes_method3d(self, index, method_name, *args, **kwargs):
        """Iterates over three axes combinations, until there is a combination for which this method is implemented.

        Args:
            index(tuple three ints): axes to look for method implementation at.
            method_name(str): method name.
            *args, **kwargs: passed to method.

        Returns:
            result of the method execution.

        Raises:
            NotImplementedError if combination is not found.
        """
        ax = [self.axes[index[i]] for i in (0, 1, 2)]
        ind_lst = list(itertools.permutations((0, 1, 2), 3))
        for p in ind_lst:
            try:
                new_axes = [ax[i] for i in p]
                func = getattr(new_axes[0], method_name)
                res = func(new_axes[1], new_axes[2], *args, **kwargs)
                if isinstance(res, Iterable):
                    res = list(res)
                    for i in range(3):
                        for j, item in enumerate(res):
                            res[j] = np.moveaxis(item, p[i], i)
                else:
                    for i in range(3):
                        res = np.moveaxis(res, p[i], i)
                return res
            except (NotImplementedError, TypeError):
                pass
        raise NotImplementedError("Custom axis should implement {} method.".format(method_name))

    def mesh_method2d(self, index, method_name, *args, **kwargs):
        """Iterates over two axes combinations, until there is a combination for which this method is implemented.

        Args:
            index(tuple three ints): axes to look for method implementation at.
            method_name(str): method name.
            *args, **kwargs: passed to method.

        Returns:
            result of the method execution.

        Raises:
            TypeError if combination is not found.
        """
        try:
            func = getattr(self.axes[index[0]], method_name)
            res = func(self.axes[index[1]], *args, **kwargs)
            return res
        except (NotImplementedError, TypeError):
            try:
                func = getattr(self.axes[index[1]], method_name)
                res = func(self.axes[index[0]], *args, **kwargs)
                if isinstance(res, Iterable):
                    for j, item in enumerate(res):
                        res[j] = item.transpose()
                else:
                    res = res.transpose()
                return res
            except (NotImplementedError, TypeError):
                raise TypeError("Custom axis should implement {} method.".format(method_name))

    def draw_mesh(self):
        pass

    def density(self, data, coordinate):
        pass
