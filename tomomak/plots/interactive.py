import matplotlib.pyplot as plt


class DetectorPlotSlider:
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
        self.slider = None
        self.data = data
        self.ax = ax

    def redraw(self):
        """Basic plot redraw function.

        Changes detector index in plot title and rescales axes.
        """
        new_title = 'Detector {}/{}'.format(self.ind + 1, self.data.shape[0])
        self.ax.set_title(new_title)
        self.ax.relim()

    def update(self, val):
        """Callback function for button Next.

        Changes self.ind and initiate redraw.
        """
        self.ind = int(val)
        self.redraw()
        plt.draw()

    def next(self, _):
        """Callback function for button Next.

        Changes self.ind and initiate redraw.
        """
        self.ind = (self.ind + 1) % self.data.shape[0]
        self.slider.set_val(self.ind)

    def prev(self, _):
        """Callback function for button Prev.

        Changes self.ind and initiate redraw.
        """
        self.ind = (self.ind - 1) % self.data.shape[0]
        self.slider.set_val(self.ind)


