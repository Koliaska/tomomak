import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button, Slider
from . import interactive


def colormesh2d(data, axis1, axis2, title='', style='colormesh', fill_scheme='viridis', grid=False, norm=None, *args,  **kwargs):
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
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.pcolormesh

     Returns:
         plot: matplotlib pcolormesh .
         ax (axes.Axes ): axes.Axes object or array of Axes objects.
             See matplotlib Axes class
         fig (matplotlib.figure): The figure module.
         cb (matplotlib.pyplot.colorbar): colorbar on the right of the axis.
     """
    cmap = plt.get_cmap(fill_scheme)
    fig, ax = plt.subplots()
    if style == 'colormesh':
        x = axis1.cell_edges1d
        y = axis2.cell_edges1d
        func = ax.pcolormesh
    elif style == 'contour':
        x = axis1.coordinates
        y = axis2.coordinates
        func = ax.contourf
    z = data.transpose()
    if norm is not None:
        plot = func(x, y, z, cmap=cmap, vmin=norm[0], vmax=norm[1], *args,  **kwargs)
    else:
        plot = func(x, y, z, cmap=cmap, *args, **kwargs)
    cb = fig.colorbar(plot, ax=ax)
    ax.set_title(title)
    xlabel = "{}, {}".format(axis1.name, axis1.units)
    ylabel = "{}, {}".format(axis2.name, axis2.units)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if grid:
        ax.grid()
    return plot, ax, fig, cb

def detector_colormesh2d(data, axis1, axis2, title='', cb_title='',  style='colormesh',  fill_scheme='viridis',
                         grid=False, equal_norm=False, *args, **kwargs):
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
            super().__init__(data_input, axis)
            self.fig = figure
            self.cb = color_bar
            self.norm = normalization
            self.slider = slider

        def redraw(self):
            y_data = np.transpose(self.data[self.ind-1])
            plot.set_array(y_data.flatten())
            if self.norm is None:
                normalization = colors.Normalize(np.min(y_data), np.max(y_data))
                self.cb.mappable.set_norm(normalization)
                self.cb.draw_all()
            super().redraw()


    norm = None
    if equal_norm:
        norm = [min(np.min(data), 0), np.max(data)]
    plot, ax, fig, cb = colormesh2d(data[0], axis1, axis2, title, style,  fill_scheme, grid, norm, *args, **kwargs)
    cb.set_label(cb_title)

    # slider
    slider_color = 'lightgoldenrodyellow'
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.12, 0.05, 0.62, 0.03], facecolor=slider_color)
    slider = Slider(ax_slider, '', 1, data.shape[0], valinit=1, valstep=1)
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

