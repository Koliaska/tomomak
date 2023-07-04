import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button, Slider
from . import interactive
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def patches(data, axis1, axis2, title='', fill_scheme='viridis', norm=None, ax_names=('-', '-'), fig_name='Figure',
            *args,  **kwargs):
    """Prepare patches plot for 2D data visualization.

    Works with arbitrary axes types.
    Args:
        data (ndarray): 2D array of data.
        axis1 (axis): corresponding tomomak axis № 1.
        axis2 (axis): corresponding tomomak axis № 2.
        title (str, optional): Plot title. default: ''.
        fill_scheme (pyplot colormap, optional): pyplot colormap to be used in the plot. default: 'viridis'.
        norm (None/[Number, Number], optional): If not None, all detectors will have same z axis
            with [ymin, ymax] = norm. default: None.
        ax_names (list of str, optional): caption for coordinate axes. Default: ('-', '-').
        fig_name (str, optional): Figure ID. Same as num parameter in matplotlib. Default: 'Figure'.
        *args, **kwargs: not used.

    Returns:
         pc: PatchCollection.
         ax (axes.Axes ): axes.Axes object or array of Axes objects.
             See matplotlib Axes class
         fig (matplotlib.figure): The figure module.
         cb (matplotlib.pyplot.colorbar): colorbar on the right of the axis.
    """
    cmap = plt.get_cmap(fill_scheme)
    fig, ax = plt.subplots(num=fig_name)
    try:
        edges = axis1.cell_edges2d_cartesian(axis2)
    except TypeError:
        edges = axis2.cell_edges2d_cartesian(axis1).transpose()
    z = data
    patch_list = []
    x_max, x_min = edges[0][0][0]
    y_max, y_min = edges[0][0][1]
    for row in edges:
        for points in row:
            polygon = Polygon(points)
            patch_list.append(polygon)
            for p in points:
                x_max = max(x_max, p[0])
                x_min = min(x_min, p[0])
                y_max = max(y_max, p[1])
                y_min = min(y_min, p[1])
    pc = PatchCollection(patch_list, alpha=1)
    pc.set_array(np.array(z).flatten())
    if norm is not None:
        pc.set_clim(norm[0], norm[1])
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, cmap=cmap)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    _set_labels(ax, title, ax_names)
    return pc, ax, fig, cb


def colormesh2d(data, axis1, axis2, title='', style='colormesh', fill_scheme='viridis',
                grid=False, norm=None, ax_names=('-', '-'), fig_name='Figure', *args,  **kwargs):
    """Prepare bar plot for 2D data visualization.

     matplotlib.pyplot.pcolormesh  is used.

     Args:
        data (ndarray): 2D array of data.
        axis1 (axis): corresponding tomomak axis № 1.
        axis2 (axis): corresponding tomomak axis № 2.
        title (str, optional): Plot title. default: ''.
        style (str, optional): Plot style. Available options: 'colormesh', 'contour'. Default: 'colormesh'.
        fill_scheme (pyplot colormap, optional): pyplot colormap to be used in the plot. default: 'viridis'.
        grid (bool, optional): if True, grid is shown. default: False.
        norm (None/[Number, Number], optional): If not None, all detectors will have same z axis
            with [ymin, ymax] = norm. default: None.
        ax_names (list of str, optional): caption for coordinate axes. Default: ('-', '-').
        fig_name (str, optional): Figure ID. Same as num parameter in matplotlib. Default: 'Figure'.
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.pcolormesh

     Returns:
         plot: matplotlib pcolormesh .
         ax (axes.Axes ): axes.Axes object or array of Axes objects.
             See matplotlib Axes class
         fig (matplotlib.figure): The figure module.
         cb (matplotlib.pyplot.colorbar): colorbar on the right of the axis.
     """
    cmap = plt.get_cmap(fill_scheme)
    fig, ax = plt.subplots(num=fig_name)
    if style == 'colormesh':
        x = axis1.cell_edges
        y = axis2.cell_edges
        func = ax.pcolormesh
        z = data.transpose()
    elif style == 'contour':
        try:
            coordinates = axis1.cartesian_coordinates(axis2)
        except TypeError:
            coordinates = axis2.cartesian_coordinates(axis1).transpose()
        x = coordinates[0].flatten()
        y = coordinates[1].flatten()
        func = ax.tricontourf
        # data = data.transpose()
        z = data.flatten()
    if norm is not None:
        plot = func(x, y, z, cmap=cmap, vmin=norm[0], vmax=norm[1], *args,  **kwargs)
    else:
        plot = func(x, y, z, cmap=cmap, *args, **kwargs)
    cb = fig.colorbar(plot, ax=ax)
    _set_labels(ax, title, ax_names)
    if grid:
        ax.grid()
    return plot, ax, fig, cb


def detector_plot2d(data, axis1, axis2, title='', cb_title='', style='colormesh', fill_scheme='viridis',
                    grid=False, equal_norm=False, transpose=True, plot_type='colormesh', ax_names=('-', '-'),
                    *args, **kwargs):
    """Prepare bar plot for 2D detector data visualization with interactive elements.

    matplotlib.pyplot.pcolormesh  is used. Interactive elements are Next and Prev buttons to change detectors.

     Args:
        data (ndarray): 2D array of data.
        axis1 (axis): corresponding tomomak axis № 1.
        axis2 (axis): corresponding tomomak axis № 2.
        title (str, optional): Plot title. Default: ''.
        style (str, optional): Plot style. Available options: 'colormesh', 'contour'. Default: 'colormesh'.
        cb_title (str, optional) Colorbar title. Default: ''.
        fill_scheme (pyplot colormap, optional): pyplot colormap to be used in the plot. Default: 'viridis'.
        grid (bool, optional): if True, grid is shown. Default: False.
        equal_norm (bool, optional): If True,  all detectors will have same z axis.
            If False, each detector has individual z axis. Default:False
        transpose (bool, optional): transpose y data for correct representation. Default: True.
        plot_type (str, optional): type of the 2d plot: 'colormesh' or 'patches'. Default: 'colormesh'.
        ax_names (list of str, optional): caption for coordinate axes. Default: ('-', '-').
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.pcolormesh

    Returns:
    plot: matplotlib bar plot.
    ax (matplotlib.axes.Axes): axes.Axes object or array of Axes objects.
        See matplotlib Axes class
    (b_next, b_prev) tuple(matplotlib.widgets.Button): Tuple, containing Next and Prev buttons.
        Objects need to exist in order to work.
         """

    class ColormeshSlider(interactive.DetectorPlotSlider):
        def __init__(self, data_input, axis, figure, color_bar, normalization, slider):
            super().__init__(data_input, axis, slider)
            self.fig = figure
            self.cb = color_bar
            self.norm = normalization

        def redraw(self):
            y_data = self.data[self.ind]
            if transpose:
                y_data = np.transpose(y_data)
            plot.set_array(y_data.flatten())
            if self.norm is None:
                normalization = colors.Normalize(np.min(y_data), np.max(y_data))
                self.cb.mappable.set_norm(normalization)
                self.cb.draw_all()
            super().redraw()

    norm = None
    if equal_norm:
        norm = [min(np.min(data), 0), np.max(data)]
    if plot_type == 'colormesh':
        plot, ax, fig, cb = colormesh2d(data[0], axis1, axis2, title, style,  fill_scheme, grid, norm, ax_names,
                                        *args, **kwargs)
    elif plot_type == 'patches':
        plot, ax, fig, cb = patches(data[0], axis1, axis2, title, fill_scheme, norm, ax_names, *args, **kwargs)
    else:
        raise ValueError('plot_type {} is unknown. See docstring for more information.'.format(plot_type))
    cb.set_label(cb_title)

    # slider
    slider_color = 'lightgoldenrodyellow'
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.12, 0.05, 0.62, 0.03], facecolor=slider_color)
    slider = Slider(ax_slider, '', 0, data.shape[0] - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)
    callback = ColormeshSlider(data, ax, fig, cb, norm, slider)
    slider.on_changed(callback.update)
    # buttons
    ax_prev = plt.axes([0.07, 0.028, 0.02, 0.075])
    ax_next = plt.axes([0.78, 0.028, 0.02, 0.075])
    b_next = Button(ax_next, '>')
    b_prev = Button(ax_prev, '<')
    b_next.on_clicked(callback.next)
    b_prev.on_clicked(callback.prev)
    return plot, ax, (slider, b_next, b_prev)


def _set_labels(ax, title, ax_names):
    ax.set_title(title)
    xlabel = ax_names[0]
    ylabel = ax_names[1]
    ax.set(xlabel=xlabel, ylabel=ylabel)
