import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from . import interactive
from matplotlib.widgets import Button, Slider


def bar1d(data, axis, title='', ylabel='', filled=True, fill_scheme='viridis', edgecolor='black',
          grid=False, norm=None, fig_name='Figure', *args,  **kwargs):
    """Prepare bar plot for 1D data visualization.

    matplotlib.pyplot.bar plot is used.

    Args:
        data (ndarray): 1D array of data.
        axis (axis): corresponding tomomak axis.
        title (str, optional): Plot title. default: ''.
        ylabel (str, optional): Plot y axis label. : ''.
        filled (bool, optional): if True, bars are filled with colors from fill_scheme according to their values.
            default: True.
        fill_scheme (pyplot colormap, optional): pyplot colormap, used if filled is true.
            Color may be used instead. default: 'viridis'.
        edgecolor (color, optional): color of the bar edges. default: 'black'.
        grid (bool, optional): if True, grid is shown. default: False.
        norm (None/[Number, Number], optional):
            If not None, all detectors will have same y axis with [ymin, ymax] = norm.иdefault: None.
        fig_name (str, optional): Figure ID. Same as num parameter in matplotlib. Default: 'Figure'.
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.bar

    Returns:
        plot: matplotlib bar plot.
        ax (axes.Axes): axes.Axes object or array of Axes objects.
            See matplotlib Axes class
    """
    fig, ax = plt.subplots(num=fig_name)
    color = None
    if filled:
        color = _fill_plot(data, fill_scheme, norm)
    ax.set(xlabel="{}, {}".format(axis.name, axis.units),
           ylabel=ylabel, title=title)
    if grid:
        ax.grid()
    plot = ax.bar(axis.coordinates, data, width=axis.volumes, color=color, edgecolor=edgecolor, *args, **kwargs)
    if norm is not None:
        plt.ylim(norm[0], norm[1])
    return plot, ax


def _fill_plot(data, fill_scheme, norm=None):
    if norm is None:
        if data.max():
            data_norm = data / data.max()
        else:
            data_norm = data
        norm = [data_norm.min(), data_norm.max()]
    else:
        data_norm = data / norm[1]
        norm = [norm[0] / norm[1], 1]
    normalyzer = colors.Normalize(norm[0], norm[1])
    color = np.zeros((len(data_norm), 4))
    fill_scheme = plt.get_cmap(fill_scheme)
    for i, v in enumerate(data_norm):
        color[i] = fill_scheme(normalyzer(v))
    return color


def detector_bar1d(data, axis, title='', ylabel='', filled=True,
                   fill_scheme='viridis', edgecolor='black', grid=False, equal_norm=False, *args, **kwargs):
    """Prepare bar plot for 1D detector data visualization with interactive elements.

    matplotlib.pyplot.bar plot is used. Interactive elements are Next and Prev buttons to change detectors.

    Args:
        data (ndarray): 1D array of data.
        axis (axis): corresponding tomomak axis.
        title (str, optional): Plot title. default: ''.
        ylabel (str, optional): Plot y axis label. : ''.
        filled (bool, optional): if True, bars are filled with colors from fill_scheme according to their values.
            default: True.
        fill_scheme (pyplot colormap, optional): pyplot colormap for to be used if filled is true or one color.
            default: 'viridis'.
        edgecolor (color, optional): color of the bar edges. default: 'black'.
        grid (bool, optional): if True, grid is shown. default: False.
        equal_norm (bool, optional): If True,  all detectors will have same y axis.
            If False, each detector has individual y axis. default:False
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.bar

    Returns:
        plot: matplotlib bar plot.
        ax (matplotlib.axes.Axes): axes.Axes object or array of Axes objects.
            See matplotlib Axes class
        (b_next, b_prev) tuple(matplotlib.widgets.Button): Tuple, containing Next and Prev buttons.
            Objects need to exist in order to work.
    """

    # Callback for Next and Prev buttons
    class BarPlotSlide(interactive.DetectorPlotSlider):
        def __init__(self, dat, ax,  normalization, slider):
            super().__init__(dat, ax, slider)
            self.norm = normalization

        def redraw(self):
            y_data = self.data[self.ind]
            for r, h in zip(plot, y_data):
                r.set_height(h)
            color = _fill_plot(y_data, fill_scheme, norm)
            for c, p in zip(color, plot):
                if filled:
                    p.set_color(c)
                p.set_edgecolor(edgecolor)
            super().redraw()
            self.ax.autoscale_view()

    norm = None
    if equal_norm:
        norm = [min(np.min(data), 0), np.max(data)*1.05]
        # normalization failed. e.g. all signals = 0
        if norm[0] == norm[1]:
            norm = None


    plot, axis = bar1d(data[0], axis, title, ylabel, filled, fill_scheme, edgecolor, grid, norm, *args, **kwargs)

    slider_color = 'lightgoldenrodyellow'
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.12, 0.05, 0.62, 0.03], facecolor=slider_color)
    slider = Slider(ax_slider, '', 0, data.shape[0] - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)
    callback = BarPlotSlide(data, axis, norm, slider)
    slider.on_changed(callback.update)
    # buttons
    ax_prev = plt.axes([0.07, 0.028, 0.02, 0.075])
    ax_next = plt.axes([0.78, 0.028, 0.02, 0.075])
    b_next = Button(ax_next, '>')
    b_prev = Button(ax_prev, '<')
    b_next.on_clicked(callback.next)
    b_prev.on_clicked(callback.prev)




    return plot, axis, (slider, b_next, b_prev)
