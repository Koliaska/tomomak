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


class ApplyAlongAxis(abstract_iterator.AbstractIterator):
    """Applies 1D function over given dimension.

    Uses numpy.apply_along_axis
    Doesn't support GPU acceleration.
    """
    def __init__(self,  func, axis=0, alpha=0.1, alpha_calc=None, **kwargs):
        """

        Args:
            func:
            axis:
            alpha:
            alpha_calc:
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
