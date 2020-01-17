from abc import ABC, abstractmethod
import warnings
import numbers
import numpy as np
import matplotlib.pyplot as plt


class AbstractSolverClass(ABC):
    """
        """

    @abstractmethod
    def init(self, model, steps, *args, **kwargs):
        """

        :return:
        """

    @abstractmethod
    def finalize(self, model):

        """

        Returns:

        """

    @abstractmethod
    def step(self, model, step_num):
        """Use this (alpha = super().step(model, step_num)) to get alpha/

        Args:
            model:
            step_num

        Returns:
            None

        """

    @abstractmethod
    def __str__(self):
        """Return name or name with parameters.

        Returns:
            str:

        """


class AbstractIterator(AbstractSolverClass):
    """
    """
    def __init__(self, alpha=0.1, alpha_calc=None):
        self.alpha = alpha
        self.alpha_calc = alpha_calc
        self._alpha = None

    @abstractmethod
    def init(self, model, steps, *args, **kwargs):
        """Use this (super().init(model, steps, *args, **kwargs)) to enable adding list of alphas.
        """
        if self.alpha_calc is not None:
            if self.alpha is not None:
                self._alpha = None
                warnings.warn("Since alpha_calc is defined in {}, alpha is Ignored.".format(self))
            self.alpha_calc.init(model, steps, *args, **kwargs)
        else:
            if isinstance(self.alpha, numbers.Number):
                self._alpha = np.full(steps, self.alpha)
            else:
                self._alpha = self.alpha
            if len(self._alpha) < steps:
                raise ValueError("Alpha len in {} should be equal or greater than number of steps.".format(self))
        """

        :return:
        """

    def get_alpha(self, model, step_num):
        """Use this to get alpha.
        """
        if self.alpha_calc is not None:
            alpha = self.alpha_calc.step(model=model)
        else:
            alpha = self._alpha[step_num]
        return alpha


class AbstractStatistics(AbstractSolverClass):
    """Abstract class for statistics calculator in solver.

    Attributes:
        data: Step-by-Step statistics data. Usually 1D Iterable.

    """

    @abstractmethod
    def step(self, *args, **kwargs):
        """Step function.

        Calculated value should be appended to self.data as well as returned.

        Args:

        Returns:
            number: statistics calculation result at current step. Needed in order to use as early stopping criteria.

        """

    def __init__(self):
        self.data = []

    def plot(self):
        plt.plot(self.data)
        plt.yscale('log')
        plt.ylabel(str(self))
