from ..iterators import abstract_iterator
import numpy as np
from tomomak.util.engine import IteratorFactory
import warnings
import numbers
from tomomak.util import array_routines
try:
    import cupy as cp
except ImportError:
    cp = None


class Tikhonov(IteratorFactory):
    """Standard 2-norm Tikhonov regularization also known as ridge regression.
    """
    pass


class _TikhonovCPU(abstract_iterator.AbstractIterator):
    def __init__(self, alpha=0.1, alpha_calc=None):
        """
        Args:
            alpha (float): weighting coefficient. The result is equal to
            old_solution + alpha * (new_solution - old_solution). Default: 0.1.
            alpha_calc (function): function for alpha calculation if needed. Default: None.
        """
        super().__init__(alpha, alpha_calc)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return "Tikhonov regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        model.solution = model.solution - alpha * model.solution / np.sqrt(np.sum(model.solution ** 2))


class _TikhonovGPU(_TikhonovCPU):
    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        model._solution = model.solution - alpha * model.solution / cp.sqrt(cp.sum(model.solution ** 2))


class Lasso(IteratorFactory):
    """Standard 1-norm regression.
    """
    pass


class _LassoCPU(abstract_iterator.AbstractIterator):
    def __init__(self, alpha=0.1, alpha_calc=None):
        """
        Args:
            alpha (float): weighting coefficient. The result is equal to
            old_solution + alpha * (new_solution - old_solution). Default: 0.1.
            alpha_calc (function): function for alpha calculation if needed. Default: None.
        """
        super().__init__(alpha, alpha_calc)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return "Lasso regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        model.solution = model.solution - alpha * np.sign(model.solution)


class _LassoGPU(_LassoCPU):
    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        model.solution = model.solution - alpha * cp.sign(model.solution)


class ElasticNet(IteratorFactory):
    """Combination of Ridge and Lasso regressions.
    """
    pass


class _ElasticNetCPU(abstract_iterator.AbstractIterator):
    def __init__(self, alpha1=0.1, alpha2=0.1, alpha_calc=None):
        """
        Args:
            alpha1 (float): weighting coefficient for L1 norm (Lasso regression)
            alpha2 (float): weighting coefficient for L2 norm (Ridge regression)
            alpha_calc (function): function for alpha calculation if needed. Default: None.
        """
        super().__init__((alpha1, alpha2), alpha_calc)

    def init(self, model, steps, *args, **kwargs):
        if self.alpha_calc is not None:
            if self.alpha is not None:
                self._alpha = None
                warnings.warn("Since alpha_calc is defined in {}, alpha is Ignored.".format(self))
            self.alpha_calc.init(model, steps, *args, **kwargs)
        else:
            if isinstance(self.alpha[0], numbers.Number):
                self._alpha = np.array((np.full(steps, self.alpha[0]), np.full(steps, self.alpha[1]))).T
            else:
                self._alpha = self.alpha
            if len(self._alpha[:, 0]) < steps:
                raise ValueError("Alpha len in {} should be equal or greater than number of steps.".format(self))

    def finalize(self, model):
        pass

    def __str__(self):
        return "Elastic Net regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        a1 = self.get_alpha(model, step_num)[0]
        a2 = self.get_alpha(model, step_num)[1]
        model.solution = model.solution - a1 * np.sign(model.solution) - a2 * model.solution / np.sqrt(
            np.sum(model.solution ** 2))


class _ElasticNetGPU(_ElasticNetCPU):

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        a1 = self.get_alpha(model, step_num)[0]
        a2 = self.get_alpha(model, step_num)[1]
        model.solution = model.solution - a1 * cp.sign(model.solution) \
                         - a2 * model.solution / cp.sqrt(cp.sum(model.solution ** 2))


class Entropy(IteratorFactory):
    """Regularization, which minimizes entropy.
    """
    pass


class _EntropyCPU(abstract_iterator.AbstractIterator):
    def __init__(self, alpha=0.1, alpha_calc=None):
        """
        Args:
            alpha (float): weighting coefficient. The result is equal to
            old_solution + alpha * (new_solution - old_solution). Default: 0.1.
            alpha_calc (function): function for alpha calculation if needed. Default: None.
        """
        super().__init__(alpha, alpha_calc)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)

    def finalize(self, model):
        pass

    def __str__(self):
        return "Entropy regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        det_geom_sum = np.sum(model.detector_geometry, axis=0)
        det_geom_sum_ln = array_routines.multiply_along_axis(model.detector_geometry, model.detector_signal, 0)
        det_geom_sum_ln = np.sum(det_geom_sum_ln, axis=0)
        model.solution = model.solution + alpha * (det_geom_sum + det_geom_sum_ln)


class _EntropyGPU(_TikhonovCPU):
    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        det_geom_sum = cp.sum(model.detector_geometry, axis=0)

        def mult_along_axis_GPU(a, b, axis):
            dim_array = np.ones((1, a.ndim), int).ravel()
            dim_array[axis] = -1
            b_reshaped = b.reshape(dim_array)
            return a * b_reshaped
        det_geom_sum_ln = mult_along_axis_GPU(model.detector_geometry, model.detector_signal, 0)
        det_geom_sum_ln = cp.sum(det_geom_sum_ln, axis=0)
        model.solution = model.solution + alpha * (det_geom_sum + det_geom_sum_ln)
