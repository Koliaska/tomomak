from . import abstract_axes
import numpy as np


class Axis1d(abstract_axes.Abstract1dAxis):
    """1D regular or irregular coordinate system.

    Main tomomak coordinate system.
    All transforms from or to other coordinate systems, plotting, etc.
    are performed using this coordinate system as mediator.
    If regular, than it is Cartesian coordinate system.
    """

    @abstract_axes.precalculated
    def cartesian_coordinates(self, *axes):
        """See description in abstract axes.
        """
        if not axes:
            return self.coordinates
        for a in axes:
            if type(a) is not Axis1d:
                raise TypeError("Cell edges with such combination of axes are not supported.")
        axes = list(axes)
        axes.insert(0, self)
        coord = [a.coordinates for a in axes]
        return np.array(np.meshgrid(*coord, indexing='ij'))

    @abstract_axes.precalculated
    def cell_edges2d_cartesian(self, axis2):
        """See description in abstract axes.
        """
        if type(axis2) is not type(self):
            raise TypeError("Cell edges with such combination of axes are not supported.")
        shape = (self.size, axis2.size)
        res = np.zeros(shape).tolist()
        edge1 = self.cell_edges
        edge2 = axis2.cell_edges
        for i, row in enumerate(res):
            for j, _ in enumerate(row):
                res[i][j] = [(edge1[i], edge2[j]), (edge1[i + 1], edge2[j]),
                             (edge1[i + 1], edge2[j + 1]), (edge1[i], edge2[j + 1])]
        return res

    @abstract_axes.precalculated
    def cell_edges3d_cartesian(self, axis2, axis3):
        """See description in abstract axes.
        """
        if type(axis2) is not Axis1d or type(axis3) is not Axis1d:
            raise TypeError("Cell edges with such combination of axes are not supported.")
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
                    faces[i][j][k] = [(0, 3, 2, 1), (4, 5, 6, 7),
                                      (2, 3, 7, 6), (4, 7, 3, 0), (0, 1, 5, 4), (1, 2, 6, 5)]
        return vertices, faces


