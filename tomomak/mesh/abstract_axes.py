from abc import ABC, abstractmethod


class AbstractAxis(ABC):
    """Superclass for every axis.
    One axis always corresponds to one data array dimension, however may corresponds to several real-world dimensions.
    If you want to create new coordinate system, inherit Abstract1DAxis, Abstract2dAxis or Abstract3dAxis.
    """

    def __init__(self, name="", units=""):
        self.name = name
        self.units = units

    @property
    @abstractmethod
    def dimension(self):
        """Get number of dimensions.

        Returns:
            int: number of dimensions
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
    def coordinates(self):
        """Get coordinates of each cell.

        Returns:
           iterable of floats or points (list of floats): coodinates if each cell centers.
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


class Abstract1dAxis(AbstractAxis):
    """Superclass for 1D axis.

    1D means that one data array dimension (solution or detector geometry) is describing one real-world dimension.
    Axes need to be stacked, e.g. six 1D cartesian axes describe 6D space in a real world.
    """
    @property
    def dimension(self):
        return 1

    @abstractmethod
    def plot1d(self, data, data_type, *args, **kwargs):
        """Create 1D plot.

        Args:
            data (ndarray): data to plot.
            data_type (str): type of plotted data: ''detectors' or 'solution'.
            *args, **kwargs: additional parameters. Depends on specific implementation.

        Returns:
            plot data, specific for each implementation.
        """

    @abstractmethod
    def plot2d(self, data, axis2, data_type, *args, **kwargs):
        """Create 2D plot.

        Args:
            data (ndarray): data to plot.
            axis2 (tomomak axis): second 1D plot axis.
            data_type (str): type of plotted data: ''detectors' or 'solution'.
            *args, **kwargs: additional parameters. Depends on specific implementation.

        Returns:
            plot data, specific for each implementation.
        """

    @abstractmethod
    def plot3d(self, data, axis2, axis3, data_type, *args, **kwargs):
        """Create 3D plot.

        Args:
            data (ndarray): data to plot.
            axis2 (tomomak axis): second 1D plot axes.
            axis3 (tomomak axis): third 1D plot axes.
            data_type (str): type of plotted data: ''detectors' or 'solution'.
            *args, **kwargs: additional parameters. Depends on specific implementation.

        Returns:
            plot data, specific for each implementation.
        """

    @property
    @abstractmethod
    def cell_edges1d(self):
        """Get edges of a cell in cartesian coordinates.

        Returns:
            1D list of points (x, y) in cartesian coordinates: points, representing the cell.
            Size of the list is self.size.
        """

    @abstractmethod
    def cell_edges2d(self, axis2):
        """Get edges of a cell on a mesh consisting of this and another 1D axis in cartesian coordinates.

        Args:
            axis2 (tomomak axis): another 1D axis of the same or other type.

        Returns:
            2D list of lists of points (x, y) in cartesian coordinates: points of the polygon, representing the cell.
            Size of the list is self.size x axis1.size
        """

    @abstractmethod
    def cell_edges3d(self, axis2, axis3):
        """Get edges of a cell on a mesh consisting of this and two other 1D axis in cartesian coordinates.

        Args:
            axis2 (tomomak axis): another 1D axes of the same or other type.
            axis3 (tomomak axis): another 1D axes of the same or other type.

        Returns:
            3D list of lists of points (x, y) in cartesian coordinates: points of the polygon, representing the cell.
            Size of the list is self.size x axis1.size
        """


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
    def cell_edges2d(self):
        """

        Args:
            axis2:

        Returns:

        """

    @abstractmethod
    def cell_edges3d(self, axis2):
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
    def cell_edges3d(self):
        """

        Args:
            axis2:

        Returns:

        """
