from . import cartesian
import numpy as np
from . import abstract_axes
from . import polar
import matplotlib.pyplot as plt
from tomomak.plots import plot2d
from tomomak import util


class Axis1d(abstract_axes.Abstract1dAxis):
    """
    From 0  to 2pi
    toroidal + polar + cartesian -> 3d toroidal system
    """

    RESOLUTION3D = 5

    def __init__(self, radius, coordinates=None, edges=None, lower_limit=0,
                 upper_limit=2*np.pi, size=None, name="", units=""):
        super().__init__(coordinates, edges, lower_limit, upper_limit, size, name, units, True)
        self._R = radius
        self._check_self_consistency()

    def _check_self_consistency(self):
        if self._R <= 0:
            raise ValueError("Radius should be > 0.")
        if np.any(self.cell_edges < 0):
            raise ValueError("Grid edges are < 0. Toroidal axis coordinates should be > 0 and < 2*pi.")
        elif np.any(self.cell_edges > 2 * np.pi):
            raise ValueError("Grid edges are > 2*pi. Toroidal axis coordinates should be > 0 and < 2*pi.")

    @property
    def radius(self):
        """See description in abstract axes.
        """
        return self._R

    def cell_edges2d_cartesian(self, axis2):
        raise AttributeError("cell_edges2d_cartesian with such combination of axes is not supported.")

    @abstract_axes.precalculated
    def cell_edges3d_cartesian(self, axis2, axis3):
        # Toroidal coordinates
        if type(axis2) is polar.Axis1d and type(axis3) is cartesian.Axis1d:
            if not axis3.spatial:
                raise ValueError("Toroidal axis works only with spatial axes")
            shape = (self.size, axis2.size, axis3.size)
            vertices = np.zeros(shape).tolist()
            faces = np.zeros(shape).tolist()
            edge_tor = self.cell_edges
            edges2d = axis2.cell_edges2d_cartesian(axis3)
            res2d = axis2.RESOLUTION2D
            # Precalculate faces for 2 cases - 1) standard, 2) - center of the mesh
            # 1) standard
            face = list()
            layer_len = res2d * 2 + 2
            # first front and back faces
            face.append((2, 1, 0))
            face.append((2, 0, res2d * 2 + 1))
            face.append((layer_len * self.RESOLUTION3D, layer_len * self.RESOLUTION3D + 1,
                         layer_len * self.RESOLUTION3D + 2))
            face.append((layer_len * self.RESOLUTION3D,
                         layer_len * self.RESOLUTION3D + 2, layer_len * (self.RESOLUTION3D + 1) - 1))
            for i in range(res2d - 1):
                # front
                face.append((i + 3, i + 2, res2d * 2 - i + 1))
                face.append((i + 3, res2d * 2 - i + 1, res2d * 2 - i))
                # back
                face.append((layer_len * self.RESOLUTION3D + 2 + i, layer_len * self.RESOLUTION3D + 3 + i,
                             layer_len * (self.RESOLUTION3D + 1) - i - 2))
                face.append((layer_len * self.RESOLUTION3D + 2 + i,
                             layer_len * (self.RESOLUTION3D + 1) - i - 2, layer_len * (self.RESOLUTION3D + 1) - i - 1))
            for l in range(self.RESOLUTION3D):
                # left and right faces
                layer = l * layer_len
                next_layer = (l + 1) * layer_len
                face.append((layer, layer + 1, next_layer + 1))
                face.append((layer, next_layer + 1, next_layer))
                face.append((layer + res2d + 1, layer + res2d + 2,
                             next_layer + res2d + 2))
                face.append((layer + res2d + 1,
                             next_layer + res2d + 2, next_layer + res2d + 1))
                # first top and bot face
                face.append((layer + 1, layer + 2, next_layer + 2))
                face.append((layer + 1, next_layer + 2, next_layer + 1))
                face.append((layer, next_layer, (l + 2) * layer_len - 1))
                face.append((layer, (l + 2) * layer_len - 1, next_layer - 1))
                for i in range(res2d - 1):
                    # top
                    face.append((layer + i + 2, layer + i + 3, layer + res2d * 2 + i + 5))
                    face.append((layer + i + 2, layer + res2d * 2 + i + 5, layer + res2d * 2 + i + 4))
                    # bottom
                    face.append((layer + res2d * 2 - i,layer + res2d * 2 - i + 1,
                                 layer + res2d * 4 - i + 3))
                    face.append((layer + res2d * 2 - i,
                                 layer + res2d * 4 - i + 3, layer + res2d * 4 - i + 2))
            # 2) - center of the mesh
            faces_center = list()
            layer_len = res2d + 2
            for l in range(self.RESOLUTION3D):
                layer = l * layer_len
                next_layer = (l + 1) * layer_len
                # left and right faces
                faces_center.append((layer, layer + 1, next_layer))
                faces_center.append((layer + 1,  next_layer + 1, next_layer))
                faces_center.append((next_layer - 1, layer, next_layer))
                faces_center.append((next_layer - 1, next_layer, (l + 2) * layer_len - 1))
                for i in range(res2d):
                    # top
                    faces_center.append((layer + i + 1, next_layer + i + 2, next_layer + i + 1))
                    faces_center.append((layer + i + 1, layer + i + 2, next_layer + i + 2))
            for i in range(res2d):
                # front
                faces_center.append((i + 2, i + 1, 0))
                # back
                faces_center.append((layer_len * self.RESOLUTION3D + 1 + i, layer_len * self.RESOLUTION3D + 2 + i,
                                     layer_len * self.RESOLUTION3D))
            for i, row in enumerate(vertices):
                tor_step = (edge_tor[i + 1] - edge_tor[i]) / self.RESOLUTION3D
                for j, col in enumerate(row):
                    for k, _ in enumerate(col):
                        vertices[i][j][k] = []
                        for l in range(self.RESOLUTION3D + 1):
                            for e in edges2d[j][k]:
                                vertices[i][j][k].append(((e[0] + self._R) * np.cos(edge_tor[i] + l * tor_step), e[1],
                                                          (e[0] + self._R) * np.sin(edge_tor[i] + l * tor_step)))
                        if len(edges2d[j][k]) == res2d * 2 + 2:
                            faces[i][j][k] = face
                        else:
                            faces[i][j][k] = faces_center
            return np.array(vertices, dtype=object), np.array(faces, dtype=object)
        else:
            raise TypeError("cell_edges3d_cartesian with such combination of axes is not supported.")

    @abstract_axes.precalculated
    def cartesian_coordinates(self, *axes):
        """See description in abstract axes.
                     """
        if not axes:
            raise ValueError("Impossible to convert 1D polar axis to cartesian coordinates.")
        if len(axes) == 2:
            shape = (self.size, axes[0].size, axes[1].size)
            x = np.zeros(shape)
            y = np.zeros(shape)
            z = np.zeros(shape)
            # Toroidal coordinates
            if type(axes[0]) is polar.Axis1d and type(axes[1]) is cartesian.Axis1d:
                if not axes[1].spatial:
                    raise ValueError("Toroidal axis works only with spatial axes")
                x2d, y2d = axes[0].cartesian_coordinates(axes[1])
                tor_axis = self.coordinates
                for i, row in enumerate(x):
                    for j, col in enumerate(row):
                        for k, _ in enumerate(col):
                            x[i, j, k] = (x2d[j, k] + self._R) * np.cos(tor_axis[i])
                            y[i, j, k] = (x2d[j, k] + self._R) * np.sin(tor_axis[i])
                            z[i, j, k] = y2d[j, k]
                return x, y, z
        raise TypeError("cartesian_coordinate with such combination of axes is not supported.")

    @staticmethod
    def _polar_to_cart(x, y):
        r = np.sqrt(x**2 + y**2)
        phi = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        return r, phi

    def from_cartesian(self, coordinates, *axes):
        if len(axes) == 2:
            xx, yy, zz = coordinates[0], coordinates[1], coordinates[2]
            if type(axes[0]) is polar.Axis1d and type(axes[1]) is cartesian.Axis1d:
                if not axes[1].spatial:
                    raise ValueError("Toroidal axis works only with spatial axes")
                r_hor, theta = self._polar_to_cart(xx, yy)
                r, phi = self._polar_to_cart((r_hor - self._R), zz)
                return theta, phi, r
        else:
            raise TypeError("from_cartesian with such combination of axes is not supported.")

    def plot3d(self, data, axis2, axis3, mesh, data_type='solution', colormap='blue-red', axes=False,
               cartesian_coordinates=False, interp_size=None, *args, **kwargs):
        if cartesian_coordinates:
            if not (axis2.spatial and axis3.spatial):
                raise ValueError("Toroidal axis works only with spatial axes for converting to toroidal coordinates.")
            if type(axis2) is polar.Axis1d and type(axis3) is cartesian.Axis1d and axis3.spatial:
                ax_names = ('{}, {}'.format('X', axis3.units),
                            '{}, {}'.format('Y', axis3.units),
                            '{}, {}'.format('Z', axis3.units))
            else:
                raise TypeError("plot3d with such combination of axes is not supported.")
        else:
            ax_names = None
        plot, ax = super().plot3d(data, axis2, axis3, mesh, data_type, colormap, axes,
                                  cartesian_coordinates, interp_size, ax_names, *args, **kwargs)

        return plot, ax
