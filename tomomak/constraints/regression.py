from ..iterators import abstract_iterator
import numpy as np
from tomomak.util.engine import IteratorFactory
import warnings
import numbers
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
        return "Tikhonov regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        alpha = self.get_alpha(model, step_num)
        model.solution = model.solution - alpha


class _LassoGPU(_LassoCPU):
    pass

class ElasticNet(IteratorFactory):
    """Combination of Ridge and Lasso regressions.
    """
    pass


class _ElasticNetCPU(abstract_iterator.AbstractIterator):

    def __init__(self, alpha1=0.1, alpha2=0.1,  alpha_calc=None):
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
        return "Tikhonov regularization"

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        a1 = self.get_alpha(model, step_num)[0]
        a2 = self.get_alpha(model, step_num)[1]
        model.solution = model.solution - a1 - a2 * model.solution / np.sqrt(np.sum(model.solution ** 2))


class _ElasticNetGPU(_ElasticNetCPU):

    def step(self, model, step_num):
        """Step function, called by solver.
        """
        a1 = self.get_alpha(model, step_num)[0]
        a2 = self.get_alpha(model, step_num)[1]
        model.solution = model.solution - a1 - a2 * model.solution / cp.sqrt(cp.sum(model.solution ** 2))