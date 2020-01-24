from ..iterators import abstract_iterator
import numpy as np


class Positive(abstract_iterator.AbstractSolverClass):
    """Makes all negative values equal to 0.
    """
    def __init__(self):
        pass

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "Remove negative values"

    def step(self, model, step_num):
        model.solution = model.solution.clip(min=0)


class ApplyAlongAxis(abstract_iterator.AbstractIterator):
    """Applies 1D function over given dimension.

    Uses numpy.apply_along_axis
    """
    def __init__(self,  func, axis=0, alpha=0.1, alpha_calc=None, **kwargs):
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


class ApplyFunction(abstract_iterator.AbstractIterator):
    """Applies multidimensional function over solution.

    Note, that function should be able to work with array of the solution dimension.
    """

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
