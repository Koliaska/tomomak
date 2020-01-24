import numpy as np
from tomomak.detectors import signal
from tomomak.iterators.abstract_iterator import AbstractStatistics


class RMS(AbstractStatistics):
    """Calculate normalized root mean square error.

    RMS is between calculated and real solution.
    Real solution should be defined in order to get RMS: to do this set real_solution member of solver object.

    """
    def step(self, solution, real_solution, *args, **kwargs):
        """Calculate a normalized root mean square error at current step

        Args:
            solution (ndarray): supposed solution.
            real_solution (ndarray): known solution.
            *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: normalized RMS.

        """
        res = solution - real_solution
        res = np.square(res)
        res = np.sum(res)
        tmp = np.sum(np.square(solution))
        if tmp != 0:
            res = np.sqrt(res / tmp) * 100
        else:
            res = float("inf")
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "RMS, %"


class RN(AbstractStatistics):
    """Calculate Residual Norm.

    RN is between calculated and measured signal.
    """

    def step(self, model,  real_solution, *args, **kwargs):
        """Residual norm at current step.

        Args:
            model (tomomak.Model): used model.
            real_solution (ndarray): known solution.
             *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: residual norm

        """
        norm = model.detector_signal - signal.get_signal(model.solution, model.detector_geometry)
        norm = np.square(norm)
        res = np.sqrt(np.sum(norm))
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "RN"


class ChiSq(AbstractStatistics):
    """Chi^2 statistics at current step.

    Chi^2 is between calculated and real solution.
    Real solution should be defined in order to get Chi^2: to do this set real_solution member of solver object.

    """

    def step(self, solution, real_solution, *args, **kwargs):
        """Chi^2 statistics.

        Note, that fo usage in feasibility method chi^2 should be divided by number of detectors.
        Args:
            solution (ndarray): supposed solution.
            real_solution (ndarray): known solution.
             *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: chi^2.

        """
        chi = solution - real_solution
        chi = chi ** 2
        chi = np.divide(chi, real_solution, out=np.zeros_like(chi), where=real_solution != 0)
        res = np.sum(chi)
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "chi-square"


class CorrCoef(AbstractStatistics):
    """Calculate correlation coefficient, used for stopping criterion.
    """
    def step(self, model, solution, old_solution, *args, **kwargs):
        """Correlation coefficient at current step.

        See Craciunescu et al., Nucl. Instr. and Meth. in Phys. Res. A595 2008 623-630.

        Args:
            model (tomomak.Model): used model.
            solution (ndarray): supposed solution.
            old_solution (ndarray): supposed_solution at a previous iteration.
             *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: correlation coefficient.
        """
        det_num = model.detector_signal.shape[0]
        det_num2 = det_num**2
        f_s = np.sum(old_solution)
        f_new_s = np.sum(solution)
        corr = det_num2 * np.sum(np.multiply(solution, old_solution))
        corr = corr - f_s * f_new_s
        divider = det_num2 * np.sum(np.multiply(solution, solution))
        tmp = f_new_s**2
        divider = np.sqrt(divider - tmp)
        corr = corr / divider
        divider = det_num2 * np.sum(np.multiply(old_solution, old_solution))
        tmp = f_s**2
        divider = np.sqrt(divider - tmp)
        res = corr / divider
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "cor. coef."


class Convergence(AbstractStatistics):
    """calculate d(solution) / solution * 100%.
    """

    def step(self, solution, old_solution, *args, **kwargs):
        """calculate d(solution) / solution * 100%.

        Args:
            solution (ndarray): supposed solution.
            old_solution (ndarray): solution at previous step
             *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: ds/s, %

        """
        res = np.sum(np.abs(solution - old_solution)) / np.abs(np.sum(solution)) * 100
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "ds/s, %"
