import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from . import interactive


def colormesh2d(data, axis1, axis2, title='', fill_scheme='viridis', grid=False, norm=None, *args,  **kwargs):
    """Prepare bar plot for 2D data visualization.

     matplotlib.pyplot.pcolormesh  is used.

     Args:
        data (ndarray): 2D array of data.
        axis1 (axis): corresponding tomomak axis № 1.
        axis2 (axis): corresponding tomomak axis № 2.
        title (str, optional): Plot title. default: ''.
        fill_scheme (pyplot colormap, optional): pyplot colormap to be used in the plot. default: 'viridis'.
        grid (bool, optional): if True, grid is shown. default: False.
        norm (None/[Number, Number], optional): If not None, all detectors will have same z axis
            with [ymin, ymax] = norm. default: None.
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.pcolormesh

     Returns:
         plot: matplotlib pcolormesh .
         ax (axes.Axes ): axes.Axes object or array of Axes objects.
             See matplotlib Axes class
         fig (matplotlib.figure): The figure module.
         cb (matplotlib.pyplot.colorbar): colorbar on the right of the axis.
     """
    x = axis1.cell_edges1d
    y = axis2.cell_edges1d
    z = data.transpose()
    cmap = plt.get_cmap(fill_scheme)
    fig, ax = plt.subplots()
    if norm is not None:
        plot = ax.pcolormesh(x, y, z, cmap=cmap, vmin=norm[0], vmax=norm[1], *args,  **kwargs)
    else:
        plot = ax.pcolormesh(x, y, z, cmap=cmap, *args, **kwargs)
    cb = fig.colorbar(plot, ax=ax)
    ax.set_title(title)
    xlabel = "{}, {}".format(axis1.name, axis1.units)
    ylabel = "{}, {}".format(axis2.name, axis2.units)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if grid:
        ax.grid()
    return plot, ax, fig, cb


def detector_colormesh2d(data, axis1, axis2, title='', fill_scheme='viridis',
                         grid=False, equal_norm=False, *args, **kwargs):
    """Prepare bar plot for 2D detector data visualization with interactive elements.

    matplotlib.pyplot.pcolormesh  is used. Interactive elements are Next and Prev buttons to change detectors.

     Args:
        data (ndarray): 2D array of data.
        axis1 (axis): corresponding tomomak axis № 1.
        axis2 (axis): corresponding tomomak axis № 2.
        title (str, optional): Plot title. default: ''.
        fill_scheme (pyplot colormap, optional): pyplot colormap to be used in the plot. default: 'viridis'.
        grid (bool, optional): if True, grid is shown. default: False.
        equal_norm (bool, optional): If True,  all detectors will have same z axis.
            If False, each detector has individual z axis. default:False
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.pcolormesh

    Returns:
    plot: matplotlib bar plot.
    ax (matplotlib.axes.Axes): axes.Axes object or array of Axes objects.
        See matplotlib Axes class
    (b_next, b_prev) tuple(matplotlib.widgets.Button): Tuple, containing Next and Prev buttons.
        Objects need to exist in order to work.
         """
    class ColormeshSlice(interactive.DetectorPlotSlicer):
        def __init__(self, data_input, axis, figure, color_bar, normalization):
            super().__init__(data_input, axis)
            self.fig = figure
            self.cb = color_bar
            self.norm = normalization

        def redraw(self):
            y_data = np.transpose(self.data[self.ind])
            plot.set_array(y_data.flatten())
            if self.norm is None:
                normalization = colors.Normalize(np.min(y_data), np.max(y_data))
                self.cb.mappable.set_norm(normalization)
                self.cb.draw_all()
            super().redraw()

    norm = None
    if equal_norm:
        norm = [min(np.min(data), 0), np.max(data)]
    plot, ax, fig, cb = colormesh2d(data[0], axis1, axis2, title,  fill_scheme, grid, norm, *args, **kwargs)
    callback = ColormeshSlice(data, ax, fig, cb, norm)
    b_next, b_prev = interactive.crete_prev_next_buttons(callback.next, callback.prev)
    return plot, ax, (b_next, b_prev)
