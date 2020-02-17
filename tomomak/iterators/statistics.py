import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from tomomak.detectors import signal
from tomomak.iterators.abstract_iterator import AbstractStatistics
from tomomak.util.engine import IteratorFactory


class RMS(IteratorFactory):
    """Calculate normalized root mean square error.

    RMS is between calculated and real solution.
    Real solution should be defined in order to get RMS: to do this set real_solution member of solver object.

    """
    pass


class _RMSCPU(AbstractStatistics):
    def step(self, solution, *args, **kwargs):
        """Calculate a normalized root mean square error at current step

        Args:
            solution (ndarray): supposed solution.
            real_solution (ndarray): known solution.
            *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: normalized RMS.

        """
        res = solution - self.real_solution
        res = np.square(res)
        res = np.sum(res)
        tmp = np.sum(np.square(solution))
        if tmp != 0:
            res = np.sqrt(res / tmp) * 100
        else:
            res = float("inf")
        self.data.append(res)
        return res

    def finalize(self, model):
        pass

    def __str__(self):
        return "RMS, %"


class _RMSGPU(_RMSCPU):
    def step(self, solution, *args, **kwargs):
        """Calculate a normalized root mean square error at current step

        Args:
            solution (ndarray): supposed solution.
            *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: normalized RMS.

        """
        self.res = solution - self.real_solution
        self.res = cp.square(self.res)
        self.res = np.sum(self.res)
        tmp = np.sum(np.square(solution))
        if tmp != 0:
            self.res = np.sqrt(self.res / tmp) * 100
        else:
            self.res = float("inf")
        self.data.append(self.res)
        return self.res

    def init(self, model, steps, real_solution, *args, **kwargs):
        real_solution = cp.array(real_solution)
        super().init(model, steps, real_solution, *args, **kwargs)
        shape = model.mesh.shape
        self.res = cp.ones(shape)

    def __init__(self):
        self.res = None
        super().__init__()

    def finalize(self, model):
        for i, d in enumerate(self.data):
            self.data[i] = cp.asnumpy(d)


class RN(IteratorFactory):
    """Calculate Residual Norm.

    RN is between calculated and measured signal.
    """
    pass


class _RNCPU(AbstractStatistics):
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

    def finalize(self, model):
        pass

    def __str__(self):
        return "RN"


class _RNGPU(_RNCPU):
    def step(self, model, *args, **kwargs):
        """Residual norm at current step.

        Args:
            model (tomomak.Model): used model.
             *args: not used, but needed to be here in order to work with Solver properly.
             **kwargs: not used, but needed to be here in order to work with Solver properly.

        Returns:
            float: residual norm

        """
        norm = model.detector_signal - signal.get_signal_gpu(model.solution, model.detector_geometry)
        norm = cp.square(norm)
        res = cp.sqrt(np.sum(norm))
        self.data.append(res)
        return res

    def finalize(self, model):
        for i, d in enumerate(self.data):
            self.data[i] = cp.asnumpy(d)


class ChiSq(IteratorFactory):
    """Chi^2 statistics at current step.

    Chi^2 is between calculated and real solution.
    Real solution should be defined in order to get Chi^2: to do this set real_solution member of solver object.

    """
    pass


class _ChiSqCPU(AbstractStatistics):
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

    def init(self, model, steps, real_solution, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "chi-square"


class _ChiSqGPU(_ChiSqCPU):
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
        chi = solution - self.real_solution
        chi = chi ** 2
        chi = cp.divide(chi, self.real_solution)
        chi = cp.where(cp.isnan(chi), 0, chi)
        res = cp.sum(chi)
        self.data.append(res)
        return res

    def init(self, model, steps, real_solution, *args, **kwargs):
        self.real_solution = cp.array(real_solution)

    def finalize(self, model):
        for i, d in enumerate(self.data):
            self.data[i] = cp.asnumpy(d)


class CorrCoef(IteratorFactory):
    """Calculate correlation coefficient, used for stopping criterion.

     See Craciunescu et al., Nucl. Instr. and Meth. in Phys. Res. A595 2008 623-630.
    """
    pass


class _CorrCoefCPU(AbstractStatistics):
    def step(self, model, solution, old_solution, *args, **kwargs):
        """Correlation coefficient at current step.

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
        if divider:
            res = corr / divider
        else:
            res = np.nan
        self.data.append(res)
        return res

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "cor. coef."


class _CorrCoefGPU(_CorrCoefCPU):
    def step(self, model, solution, old_solution, *args, **kwargs):
        """Correlation coefficient at current step.

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
        f_s = cp.sum(old_solution)
        f_new_s = cp.sum(solution)
        corr = det_num2 * cp.sum(np.multiply(solution, old_solution))
        corr = corr - f_s * f_new_s
        divider = det_num2 * cp.sum(np.multiply(solution, solution))
        tmp = f_new_s**2
        divider = cp.sqrt(divider - tmp)
        corr = corr / divider
        divider = det_num2 * cp.sum(cp.multiply(old_solution, old_solution))
        tmp = f_s**2
        divider = cp.sqrt(divider - tmp)
        if divider:
            res = corr / divider
        else:
            res = np.nan
        self.data.append(res)
        self.data.append(res)
        return res

    def finalize(self, model):
        for i, d in enumerate(self.data):
            self.data[i] = cp.asnumpy(d)
        # remove all Nones


class Convergence(IteratorFactory):
    """calculate d(solution) / solution * 100%.
    """
    pass


class _ConvergenceCPU(AbstractStatistics):
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


class _ConvergenceGPU(_ConvergenceCPU):
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
        res = cp.sum(cp.abs(solution - old_solution)) / cp.abs(cp.sum(solution)) * 100
        self.data.append(res)
        return res

    def finalize(self, model):
        for i, d in enumerate(self.data):
            self.data[i] = cp.asnumpy(d)

