import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class DetectorPlotSlicer:
    """Base class for callback function of Next and Prev buttons on detector plot.

        In order to add more functionality new class, inheriting this class, should be implementsd.
        See examples. e.g. in plot1d.detector_bar1d.

        Attributes:
            ind (int): Index of currently viewed detector.
            data (ndarray): Plotted data.
                In base class only shape of data is used.
            ax (matplotlib.axes.Axes): Plot axis.

        """
    def __init__(self, data, ax):
        """Class constructor, which requires only data and plot axes.

        Args:
            data (ndarray): Plotted data.
            ax (matplotlib.axes.Axes): Plot axes.
        """
        self.ind = 0
        self.data = data
        self.ax = ax

    def redraw(self):
        """Basic plot redraw function.

        Changes detector index in plot title and rescales axes.
        """
        new_title = 'Detector {}/{}'.format(self.ind + 1, self.data.shape[0])
        self.ax.set_title(new_title)
        self.ax.relim()


    def next(self, _):
        """Callback function for button Next.

        Changes self.ind and initiate redraw.
        """
        self.ind = (self.ind + 1) % self.data.shape[0]
        self.redraw()
        plt.draw()

    def prev(self, _):
        """Callback function for button Prev.

        Changes self.ind and initiate redraw.
        """
        self.ind = (self.ind - 1) % self.data.shape[0]
        self.redraw()
        plt.draw()


def crete_prev_next_buttons(callback_next, callback_prev):
    """Routine to create Next and Prev buttons on the interactive detector plot.

    Buttons are created using matplotlib.widgets.Button.

    Args:
        callback_next (func): Callback for the Next button.
        callback_prev (func): Callback for the Prev button.

    Returns:
        b_next: button Next
        b_prev: button Prev
    """
    plt.subplots_adjust(bottom=0.2)
    ax_prev = plt.axes([0.7, 0.02, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.02, 0.1, 0.075])
    b_next = Button(ax_next, 'Next')
    b_prev = Button(ax_prev, 'Previous')
    b_next.on_clicked(callback_next)
    b_prev.on_clicked(callback_prev)
    return b_next, b_prev
