import numpy as np
from tomomak.mesh import mesh
import copy


class Rescale:
    """
    Rescales to the new shape. Keeps proportions for irregular grid. Doesn't smooth.
    Axes need to implement constructor with 'edges' parameter.
    """

    def __init__(self, new_shape):
        self.new_shape = new_shape
        self.old_shape = None

    @staticmethod
    def _rescale(new_shape, model):
        if model.mesh is None:
            raise ValueError("Unable to rescale model with undefined Mesh.")
        if len(model.mesh.axes) != len(new_shape):
            raise ValueError("Number of the new shape axes should be equal to number of Mesh axes. "
                             "New shape has {} axes. Mesh has {} axes".format(len(new_shape), len(model.mesh.axes)))
        new_mesh = mesh.Mesh()
        # find edges of the new axes cells; create new axes; create new mesh
        for i, ax in enumerate(model.mesh.axes):
            new_len = new_shape[i]
            old_len = ax.size
            if new_len != old_len:
                ratio = old_len / new_len
                old_edges = ax.cell_edges
                new_edges = np.zeros(new_len + 1)
                new_edges[0] = old_edges[0]
                int_part = 0
                reminder = 0
                for j in range(1, new_len):
                    new_edge = int_part + reminder + ratio
                    int_part = int(new_edge // 1)
                    reminder = new_edge % 1
                    new_edges[j] = old_edges[int_part] + (old_edges[int_part + 1] - old_edges[int_part]) * reminder
                new_edges[-1] = old_edges[-1]
                new_axis = type(ax)(name=ax.name, units=ax.units, edges=new_edges)
                new_mesh.add_axis(axis=new_axis)
            else:
                new_axis = copy.copy(ax)
                new_mesh.add_axis(axis=new_axis)
        # change solution and detector_geometry.
        new_mesh = NewMesh(new_mesh)
        new_mesh(model)

    def __call__(self, model):
        old_shape = model.shape
        self._rescale(self.new_shape, model)
        self.old_shape = self.new_shape
        self.new_shape = old_shape


class NewMesh:
    def __init__(self, new_mesh):
        self.new_mesh = new_mesh
        self.old_mesh = None

    @staticmethod
    def _new_mesh(new_mesh, model, data, data_type):
        new_solution = data
        for i, ax in enumerate(model.mesh.axes):
            solution = new_solution
            new_shape = list(solution.shape)
            new_shape[i] = new_mesh.shape[i]
            new_solution = np.zeros(new_shape)
            new_solution = np.swapaxes(new_solution, 0, i)
            solution = np.swapaxes(solution, 0, i)
            try:
                inters_len = ax.intersection(new_mesh.axes[i])
                inters_len = np.transpose(inters_len)
            except (TypeError, AttributeError):
                inters_len = new_mesh.axes[i].intersection(ax)
            if data_type == 'solution':
                divider = new_mesh.axes[i].volumes
            elif data_type == 'detector_geometry':
                divider = ax.volumes  # np.ones(new_shape[i])
            else:
                raise ValueError("data type {} is unknown".format(data_type))
            for j in range(new_shape[i]):
                for k in range(solution.shape[0]):
                    if data_type == 'solution':
                        new_solution[j, ...] += solution[k, ...] * inters_len[j, k] / divider[j]
                    else:
                        new_solution[j, ...] += solution[k, ...] * inters_len[j, k] / divider[k]
            new_solution = np.swapaxes(new_solution, 0, i)
        return new_solution

    def __call__(self, model):
        old_mesh = model.mesh
        solution = model.solution
        detector_geometry = model.detector_geometry
        model.solution = None
        model.detector_geometry = None
        if solution is not None:
            solution = self._new_mesh(self.new_mesh, model, solution, 'solution')
        if detector_geometry is not None:
            new_shape = [detector_geometry.shape[0]]
            new_shape.extend(self.new_mesh.shape)
            new_detector_geometry = np.zeros(new_shape)
            for i, geom in enumerate(detector_geometry):
                new_detector_geometry[i] = self._new_mesh(self.new_mesh, model, geom, 'detector_geometry')
        else:
            new_detector_geometry = None
        model.mesh = self.new_mesh
        model.solution = solution
        model.detector_geometry = new_detector_geometry
        self.old_mesh = self.new_mesh
        self.new_mesh = old_mesh
