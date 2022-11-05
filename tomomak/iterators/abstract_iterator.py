from abc import ABC, abstractmethod
import warnings
import numbers
import numpy as np
import matplotlib.pyplot as plt


class AbstractSolverClass(ABC):
    """Abstract class for Solvers and constraints.

    Every solver or constraint should implement these four methods to work correctly.
    However, often only step() and __str__() methods are required. So init() and finalize() methods may just pass.

    If your iterator has GPU and CPU versions, than your should be a factory, returning needed instance.
    To do so use util.engine.IteratorFactory as superclass for your class. Don't add any functionality - just pass.
    Let's assume your iterator has name *iterator_name*.
    Make CPU and GPU versions of the iterator with names _*iterator_name*CPU and _*iterator_name*GPU respectively.
    See iterators.ml for example.
    """

    @abstractmethod
    def init(self, model, steps, *args, **kwargs):
        """Initialize method called by solver.

        This method is called by solver before the calculation start.
        May be used to prepare some data used in step() method.
        Args:
            model (tomomak Model): Model to work with.
            steps (int): Number of steps. May be used to check,
                if you have enough data to iterate over this amount of steps, see AbstractIterator for example.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

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
        self.real_solution = None

    def init(self, model, steps, real_solution, *args, **kwargs):
        self.real_solution = real_solution

    def plot(self):
        plt.plot(self.data)
        plt.yscale('log')
        plt.ylabel(str(self))
