from . import cartesian
import numpy as np
import re
import matplotlib.pyplot as plt
from tomomak.plots import plot1d, plot2d, plot3d
import warnings
from tomomak import util


class Axis1d(cartesian.Axis1d):
    RESOLUTION2D = 10

    def __init__(self, coordinates=None, edges=None, lower_limit=0, upper_limit=2*np.pi, size=None, name="", units=""):
        super().__init__(coordinates, edges, lower_limit, upper_limit, size, name, units)
        self._check_self_consistency()

    def _check_self_consistency(self):
        if np.any(self.cell_edges < 0):
            raise ValueError("Grid edges are < 0. Polar coordinates should be > 0 and < 2*pi.")
        elif np.any(self.cell_edges > 2*np.pi):
            raise ValueError("Grid edges are > 2*pi. Polar coordinates should be > 0 and < 2*pi.")

    def cell_edges2d(self, axis2):
        # 2D polar coordinate system
        if type(axis2) is cartesian.Axis1d:
            shape = (self.size, axis2.size)
            res = np.zeros(shape).tolist()
            edge_p = self.cell_edges
            edge_r = axis2.cell_edges
            for i, row in enumerate(res):
                for j, _ in enumerate(row):
                    points = [(edge_r[j], edge_p[i]), (edge_r[j + 1], edge_p[i])]
                    p_step = (edge_p[i + 1] - edge_p[i]) / self.RESOLUTION2D
                    for k in range(self.RESOLUTION2D):
                        points.append((edge_r[j + 1], edge_p[i] + k * p_step))
                    points.append((edge_r[j + 1], edge_p[i + 1]))
                    if edge_r[j] > 0:
                        points.append((edge_r[j], edge_p[i + 1]))
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

    def cell_edges3d(self, axis2, axis3):  #########
        pass

    @staticmethod
    def _to_cartesian2d(r, phi):
        """r, phi to x, y"""
        return r * np.cos(phi), r * np.sin(phi)

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
        raise TypeError("cartesian_coordinate with such combination of axes are not supported.")

    def plot2d(self, axis2, data,  mesh, data_type='solution', style='colormesh', fill_scheme='viridis',
               cartesian_coordinates=False, grid=False, equal_norm=False, title=None, *args, **kwargs):
        if cartesian_coordinates:
            if type(axis2) is not cartesian.Axis1d:
                raise TypeError("2D plots in cartesian coordinates"
                                          " with such combination of axes are not supported.")
            old_name = (self.name, axis2.name)
            old_units = (self.units, axis2.units)
            self.name, axis2.name = 'X', 'Y'
            self.units = axis2.units
            if style == 'colormesh':
                if data_type == 'solution':
                    if title is None:
                        units = util.text.density_units([self.units, axis2.units])
                        title = r"Density, {}".format(units)
                    plot, ax, _, _ = plot2d.patches(data, self, axis2, title, style, fill_scheme, grid, equal_norm,
                                                    *args, **kwargs)
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
            plot, ax = super().plot2d(axis2, data,  mesh, data_type, style, fill_scheme, grid, equal_norm, title, *args, **kwargs)
        return plot, ax

    def plot3d(self, data, axis2, axis3, data_type, *args, **kwargs):
        pass