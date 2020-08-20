import numbers
import pickle
import numpy as np


class Model:
    """
    Main TOMOMAK structure.
    1 axis = 1 solution array dimension
    """

    def __init__(self, detector_geometry=None, detector_signal=None, solution=None, mesh=None):
        self._detector_geometry = detector_geometry
        self._detector_signal = detector_signal
        self._solution = solution
        self._mesh = mesh
        self._check_self_consistency()

    @property
    def shape(self):
        if self.detector_geometry is not None:
            shape = self.detector_geometry[0].shape
        elif self._solution is not None:
            shape = self._solution.shape
        elif self._mesh is not None:
            shape = []
            for n in self._mesh.axes:
                shape. append(n.size)
        else:
            shape = None
        return tuple(shape)

    @property
    def size(self):
        shape = self.shape
        if shape is not None:
            size = 1
            for s in shape:
                size *= s
        else:
            size = None
        return size

    def __str__(self):
        notdef = "Not defined."
        res = "Model description:\nNumber of detectors: "
        if self._detector_signal is not None:
            n_det = str(len(self._detector_signal))
        elif self._detector_geometry is not None:
            n_det = str(self._detector_geometry.shape[0])
        else:
            n_det = notdef
        res += n_det
        n_cells = self.size
        if n_cells is None:
            n_cells = notdef
        else:
            n_cells = str(n_cells)
        res += "\nNumber of cells: {}\nMesh:\n".format(n_cells)
        if self._mesh is not None:
            mesh = str(self._mesh)
        else:
            mesh = notdef
        res += mesh
        res += "Solution: "
        if self.solution is not None:
            solution = "Defined"
        else:
            solution = notdef
        res += solution
        return res

    @property
    def detector_geometry(self):
        return self._detector_geometry

    @detector_geometry.setter
    def detector_geometry(self, value):
        self._detector_geometry = value
        self._check_self_consistency()

    def add_detector(self, det):
        self._detector_geometry = np.append(self._detector_geometry, det, axis=0)
        self._check_self_consistency()

    @property
    def detector_signal(self):
        return self._detector_signal

    @detector_signal.setter
    def detector_signal(self, value):
        self._detector_signal = value
        self._check_self_consistency()

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, value):
        self._solution = value
        self._check_self_consistency()

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self._check_self_consistency()

    def _check_self_consistency(self):
        """
        Check model self-consistency.
        Self-consistency is checked if an attribute is changed.
        """
        if self._detector_geometry is not None:
            geometry_len = len(self._detector_geometry)
            if self._detector_signal is not None:
                if not isinstance(self._detector_signal[0], numbers.Number):
                    raise TypeError("detector_signal should be 1D iterable of numbers")
                signal_len = len(self._detector_signal)
                if geometry_len != signal_len:
                    raise Exception("detector_signal and detector_geometry should have same length. "
                                    "detector_geometry len is {}; detector signal len is {}."
                                    .format(geometry_len, signal_len))
            if self._solution is not None:
                if self._solution.shape != self._detector_geometry[0].shape:
                    raise Exception("Each slice in detector_geometry should have same shape as solution. "
                                    "detector_geometry[0] shape is {}; solution shape is {}."
                                    .format(self._detector_geometry[0].shape, self._solution.shape))
        if self._mesh is not None:
            def check_shapes(val, name):
                if self.mesh.shape != val.shape:
                    raise Exception("mesh shape is inconsistent with {}. mesh shape is {} while {} is {}."
                                    .format(name, self.mesh.shape, name, val.shape))
            if self.detector_geometry is not None:
                val = self.detector_geometry[0]
                name = "detector_geometry"
                check_shapes(val, name)
            if self.solution is not None:
                val = self.solution
                name = "solution"
                check_shapes(val, name)

    def plot1d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            if self._solution is None:
                raise Exception("Solution is not defined.")
            data = self._solution
        elif data_type == "detector_geometry":
            if self._detector_geometry is None:
                raise Exception("detector_geometry is not defined.")
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plot = self._mesh.plot1d(data, index, data_type,  **kwargs)
        return plot

    def plot2d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            if self._solution is None:
                raise Exception("Solution is not defined.")
            data = self._solution
        elif data_type == "detector_geometry":
            if self._detector_geometry is None:
                raise Exception("detector_geometry is not defined.")
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plot = self._mesh.plot2d(data, index, data_type,  **kwargs)
        return plot

    def plot3d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            if self._solution is None:
                raise Exception("Solution is not defined.")
            data = self._solution
        elif data_type == "detector_geometry":
            if self._detector_geometry is None:
                raise Exception("detector_geometry is not defined.")
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plot = self._mesh.plot3d(data, index, data_type,  **kwargs)
        return plot

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fn):
        return load_model(fn)


def load_model(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)
