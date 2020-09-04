from . import cartesian
import numpy as np
from . import abstract_axes
import matplotlib.pyplot as plt
from tomomak.plots import plot2d
from tomomak import util


class Axis1d(abstract_axes.Abstract1dAxis):
    """
    From 0  to 2pi
    polar + cartesian -> 2d polar
    polar + cartesian + cartesian -> 3d cylindrical
    """
    RESOLUTION2D = 10

    def __init__(self, coordinates=None, edges=None, lower_limit=0, upper_limit=2*np.pi, size=None, name="", units=""):
        super().__init__(coordinates, edges, lower_limit, upper_limit, size, name, units)
        self._check_self_consistency()

    def _check_self_consistency(self):
        if np.any(self.cell_edges < 0):
            raise ValueError("Grid edges are < 0. Polar coordinates should be > 0 and < 2*pi.")
        elif np.any(self.cell_edges > 2 * np.pi):
            raise ValueError("Grid edges are > 2*pi. Polar coordinates should be > 0 and < 2*pi.")

    @abstract_axes.precalculated
    def cell_edges2d_cartesian(self, axis2):
        # 2D polar coordinate system
        if type(axis2) is cartesian.Axis1d:
            shape = (self.size, axis2.size)
            res = np.zeros(shape).tolist()
            edge_p = self.cell_edges
            edge_r = axis2.cell_edges
            for i, row in enumerate(res):
                for j, _ in enumerate(row):
                    points = [(edge_r[j], edge_p[i])]
                    p_step = (edge_p[i + 1] - edge_p[i]) / self.RESOLUTION2D
                    for k in range(self.RESOLUTION2D):
                        points.append((edge_r[j + 1], edge_p[i] + k * p_step))
                    points.append((edge_r[j + 1], edge_p[i + 1]))
                    if edge_r[j] > 0:
                        for k in range(self.RESOLUTION2D):
                            points.append((edge_r[j], edge_p[i + 1] - k * p_step))
                    for k, p in enumerate(points):
                        r = p[0]
                        phi = p[1]
                        points[k] = self._to_cartesian2d(r, phi)
                    res[i][j] = points
            return res
        else:
            raise TypeError("Cell edges with such combination of axes are not supported.")

    @abstract_axes.precalculated
    def cell_edges3d_cartesian(self, axis2, axis3):
        # Cylindrical coordinates
        if type(axis2) is cartesian.Axis1d and type(axis3) is cartesian.Axis1d:
            shape = (self.size, axis2.size, axis3.size)
            vertices = np.zeros(shape).tolist()
            faces = np.zeros(shape).tolist()
            edge3 = axis3.cell_edges
            edges2d = self.cell_edges2d_cartesian(axis2)
            # Precalculate faces for 2 cases - 1) standard, 2) - center of the mesh
            # 1) standart
            face = list()
            # left and right faces
            face.append((0, 1, self.RESOLUTION2D * 2 + 3, self.RESOLUTION2D * 2 + 2))
            face.append((self.RESOLUTION2D + 1, self.RESOLUTION2D + 2,
                         self.RESOLUTION2D * 3 + 4, self.RESOLUTION2D * 3 + 3))
            # first front and back faces
            face.append((2, 1, 0, self.RESOLUTION2D * 2 + 1))
            face.append((self.RESOLUTION2D * 2 + 2, self.RESOLUTION2D * 2 + 3,
                         self.RESOLUTION2D * 2 + 4, self.RESOLUTION2D * 4 + 3))
            # first top and bot face
            face.append((1, 2, self.RESOLUTION2D * 2 + 4, self.RESOLUTION2D * 2 + 3))
            face.append((0, self.RESOLUTION2D * 2 + 2, self.RESOLUTION2D * 4 + 3, self.RESOLUTION2D * 2 + 1))
            for i in range(self.RESOLUTION2D - 1):
                # front
                face.append((i + 3, i + 2, self.RESOLUTION2D * 2 - i + 1, self.RESOLUTION2D * 2 - i))
                # back
                face.append((self.RESOLUTION2D * 2 + 4 + i, self.RESOLUTION2D * 2 + 5 + i,
                             self.RESOLUTION2D * 4 + 2 - i, self.RESOLUTION2D * 4 - i + 3))
                # top
                face.append((i + 2, i + 3, self.RESOLUTION2D * 2 + i + 5, self.RESOLUTION2D * 2 + i + 4))
                # bottom
                face.append((self.RESOLUTION2D * 2 - i, self.RESOLUTION2D * 2 - i + 1,
                             self.RESOLUTION2D * 4 - i + 3, self.RESOLUTION2D * 4 - i + 2))
            # 2) - center of the mesh
            faces_center = list()
            # left and right faces as triangles
            faces_center.append((0, 1, self.RESOLUTION2D + 2))
            faces_center.append((self.RESOLUTION2D + 2, 1, self.RESOLUTION2D + 3))
            faces_center.append((self.RESOLUTION2D + 1, 0, self.RESOLUTION2D + 2))
            faces_center.append((self.RESOLUTION2D + 1, self.RESOLUTION2D + 2, self.RESOLUTION2D * 2 + 3))
            for i in range(self.RESOLUTION2D):
                # front
                faces_center.append((i + 2, i + 1, 0))
                # back
                faces_center.append((self.RESOLUTION2D + 3 + i, self.RESOLUTION2D + 4 + i, self.RESOLUTION2D + 2))
                # top as triangles
                faces_center.append((i + 1, self.RESOLUTION2D + i + 4, self.RESOLUTION2D + i + 3))
                faces_center.append((i + 2, self.RESOLUTION2D + i + 4, i + 1))
            for i, row in enumerate(vertices):
                for j, col in enumerate(row):
                    for k, _ in enumerate(col):
                        vertices[i][j][k] = []
                        for e in edges2d[i][j]:
                            vertices[i][j][k].append((e[0], e[1], edge3[k]))
                        for e in edges2d[i][j]:
                            vertices[i][j][k].append((e[0], e[1], edge3[k + 1]))
                        if len(edges2d[i][j]) == self.RESOLUTION2D * 2 + 2:
                            faces[i][j][k] = face
                        else:
                            faces[i][j][k] = faces_center
            return np.array(vertices, dtype=object), np.array(faces, dtype=object)
        else:
            raise TypeError("Cell edges with such combination of axes are not supported.")

    @staticmethod
    def _to_cartesian2d(r, phi):
        """r, phi to x, y"""
        return r * np.cos(phi), r * np.sin(phi)

    @abstract_axes.precalculated
    def cartesian_coordinates(self, *axes):
        """See description in abstract axes.
             """
        if not axes:
            raise ValueError("Impossible to convert 1D polar axis to cartesian coordinates.")
        if len(axes) == 1:
            # 2D polar coordinate system
            shape = (self.size, axes[0].size)
            x = np.zeros(shape)
            y = np.zeros(shape)
            if type(axes[0]) is cartesian.Axis1d:
                phi_ar = self.coordinates
                r_ar = axes[0].coordinates
                for i, row in enumerate(x):
                    for j, _ in enumerate(row):
                        x[i, j], y[i, j] = self._to_cartesian2d(r_ar[j], phi_ar[i])
                return x, y
        if len(axes) == 2:
            shape = (self.size, axes[0].size, axes[1].size)
            x = np.zeros(shape)
            y = np.zeros(shape)
            z = np.zeros(shape)
            # Cylindrical coordinates
            if type(axes[0]) is cartesian.Axis1d and type(axes[1]) is cartesian.Axis1d:
                x2d, y2d = self.cartesian_coordinates(axes[0])
                z_axis = axes[1].coordinates
                for i, row in enumerate(x):
                    for j, col in enumerate(row):
                        for k, _ in enumerate(col):
                            x[i, j, k] = x2d[i, j]
                            y[i, j, k] = y2d[i, j]
                            z[i, j, k] = z_axis[k]
                return x, y, z
        raise TypeError("cartesian_coordinate with such combination of axes are not supported.")

    @staticmethod
    def _polar_to_cart(x, y):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(x, y) % (2 * np.pi)
        return r, phi

    def from_cartesian(self, coordinates, *axes):
        if len(axes) == 2:
            xx, yy, zz = coordinates[0], coordinates[1], coordinates[2]
            # Cylindrical coordinates
            if type(axes[0]) is cartesian.Axis1d and type(axes[1]) is cartesian.Axis1d:
                z = zz
                r, phi = self._polar_to_cart(xx, yy)
                return phi, r, z
        else:
            raise TypeError("from_cartesian with such combination of axes are not supported.")

    def plot2d(self, axis2, data,  mesh, data_type='solution', style='colormesh', fill_scheme='viridis',
               cartesian_coordinates=False, grid=False, equal_norm=False, title=None, *args, **kwargs):
        if cartesian_coordinates:
            if type(axis2) is not cartesian.Axis1d:
                raise TypeError("2D plots in cartesian coordinates with such combination of axes are not supported.")
            old_name = (self.name, axis2.name)
            old_units = (self.units, axis2.units)
            self.name, axis2.name = 'X', 'Y'
            self.units = axis2.units
            if style == 'colormesh':
                if data_type == 'solution':
                    if title is None:
                        units = util.text.density_units([self.units, axis2.units])
                        title = r"Density, {}".format(units)
                    plot, ax, _, _ = plot2d.patches(data, self, axis2, title, fill_scheme, *args, **kwargs)
                elif data_type == 'detector_geometry':
                    title = 'Detector 1/{}'.format(data.shape[0])
                    cb_title = util.text.detector_caption(mesh)
                    plot, ax, _ = plot2d.detector_plot2d(data, self, axis2, title, cb_title, style, fill_scheme,
                                                         grid, equal_norm, False, 'patches', *args, **kwargs)
                else:
                    raise ValueError('data type {} is unknown'.format(data_type))
                plt.show()
            else:
                plot, ax = super().plot2d(axis2, data, mesh, data_type, style,
                                          fill_scheme, grid, equal_norm, title, *args, **kwargs)
            self.name, axis2.name = old_name
            self.units, axis2.units = old_units
        else:
            plot, ax = super().plot2d(axis2, data,  mesh, data_type, style,
                                      fill_scheme, grid, equal_norm, title, *args, **kwargs)
        return plot, ax

    def plot3d(self, data, axis2, axis3, mesh, data_type='solution', colormap='blue-red', axes=False,
               cartesian_coordinates=False, interp_size=None, *args, **kwargs):
        if cartesian_coordinates:
            old_name = (self.name, axis2.name, axis3.name)
            old_units = (self.units, axis2.units, axis3.units)
            self.name, axis2.name, axis3.name = 'X', 'Y', 'Z'
            if type(axis2) == cartesian.Axis1d:
                self.units = axis2.units
                axis3.units = axis2.units
            elif type(axis3) == cartesian.Axis1d:
                self.units = axis3.units
                axis2.units = axis3.units
            else:
                raise TypeError("Unable to determine axes units while converting to cartesian coordinates")
        plot, ax = super().plot3d(data, axis2, axis3, mesh, data_type, colormap, axes,
                                  cartesian_coordinates, interp_size, *args, **kwargs)
        if cartesian_coordinates:
            self.name, axis2.name, axis3.name = old_name
            self.units, axis2.units, axis3.units = old_units
        return plot, ax
