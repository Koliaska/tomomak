"""Basic penalties and constraints used for regularization."""

from ..iterators import abstract_iterator
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from tomomak.util.engine import IteratorFactory


class Positive(IteratorFactory):
    """Makes all negative values equal to 0.
    """
    pass


class _PositiveCPU(abstract_iterator.AbstractIterator):

    def __init__(self):
        super().__init__()
        pass

    def init(self, model, steps, *args, **kwargs):
        """Initialize method called by solver.

        For this class: pass.
        """
        pass

    def finalize(self, model):
        """Finalize method called by solver.

        For this class: pass.
        """
        pass

    def __str__(self):
        return "Remove negative values"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        model.solution = model.solution.clip(min=0)


class _PositiveGPU(_PositiveCPU):

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        model._solution = cp.clip(model.solution, a_min=0)


class SolutionBoundary(IteratorFactory):
    """Specify axis cooridunates outside which solution is 0 (or other predefined value).

    Doesn't support GPU acceleration.
    """
    pass


class _SolutionBoundaryCPU(abstract_iterator.AbstractIterator):

    def __init__(self, axis=0, boundaries=(np.NINF, np.inf), alpha=0.1, alpha_calc=None, outer_value=0):
        """
        Args:
            axis (int): axis number. Default: 0.
            boundaries (float, float): left and right boundary. Default: (np.NINF, np.inf).
            alpha (float): weighting coefficient. The result is equal to
            old_solution + alpha * (new_solution - old_solution). Default: 0.1.
            alpha_calc (function): function for alpha calculation if needed. Default: None.
            outer_value (float): outer value. Default: 0
        """
        super().__init__(alpha, alpha_calc)
        self.axis = axis
        self.boundaries = boundaries
        self.outer_value = outer_value

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return f"Set values outside of {self.boundaries} at axis {self.axis} equal to {self.outer_value}."

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        current_axis = model.mesh.axes[self.axis]
        l_edges = current_axis.find_position(self.boundaries[0])
        l_edge = l_edges[0]
        if l_edge < 0:
            l_edge = 0
        r_edges = current_axis.find_position(self.boundaries[1])
        r_edge = r_edges[1]

        if r_edge < 0:  # outside of grid, see find_position()
            r_edge = current_axis.size - 1
        new_solution = np.array(model.solution)
        new_solution = np.moveaxis(new_solution, self.axis, 0)
        new_solution[:l_edge] = self.outer_value
        new_solution[r_edge:] = self.outer_value
        new_solution = np.moveaxis(new_solution, 0, self.axis)
        model.solution = model.solution + alpha * (new_solution - model.solution)


class _SolutionBoundaryGPU(_SolutionBoundaryCPU):
    pass


class ApplyAlongAxis(abstract_iterator.AbstractIterator):
    """Applies 1D function over given dimension.

    Uses numpy.apply_along_axis
    Doesn't support GPU acceleration.
    """
    def __init__(self,  func, axis=0, alpha=0.1, alpha_calc=None, **kwargs):
        """

        Args:
            func (function): applied function.
            axis (int): axis number. Default: 0.
            alpha (float): weighting coefficient. The result is equal to
            old_solution + alpha * (new_solution - old_solution). Default: 0.1.
            alpha_calc (function): function for alpha calculation if needed. Default: None.
            **kwargs:
        """
        super().__init__(alpha, alpha_calc)
        self.func = func
        self.axis = axis
        self.arg_dict = {}
        self.arg_dict.update(kwargs)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return "Apply 1d function {} to axis {}.".format(self.func.__name__, self.axis)

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        new_solution = np.apply_along_axis(self.func, self.axis, model.solution, **self.arg_dict)
        model.solution = model.solution + alpha * (new_solution - model.solution)


class ApplyFunction(IteratorFactory):
    """Applies multidimensional function over solution.

    Note, that function should be able to work with array of the solution dimension.
    """
    pass


class _ApplyFunctionCPU(abstract_iterator.AbstractIterator):
    def __init__(self,  func, alpha=0.1, alpha_calc=None, **kwargs):
        super().__init__(alpha, alpha_calc)
        self.func = func
        self.arg_dict = {}
        self.arg_dict.update(kwargs)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return "Apply function {}.".format(self.func.__name__)

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        new_solution = self.func(model.solution, **self.arg_dict)
        model.solution = model.solution + alpha * (new_solution - model.solution)


class _ApplyFunctionGPU(_ApplyFunctionCPU):
    pass
