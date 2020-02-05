from . import abstract_iterator
import numpy as np
from tomomak.detectors import signal
from tomomak.util.engine import IteratorFactory
import cupy as cp


class ART(IteratorFactory):
    """A set of iterative algebraic algorithms for image reconstruction

    see E.F. Oliveira et. al., "Comparison among tomographic reconstruction algorithms with limited data".
    in the case of ART correction is applied after calculations of single ray
    GPU acceleration is supported. Requires cupy to be installed.
    To turn it on run script with environmental variable TM_GPU set to "1"
    Or just write in your script:
       import os
       os.environ["TM_GPU"] = "1"
    """
    pass


class _ARTCPU(abstract_iterator.AbstractIterator):

    iter_types = ('ART', 'MART')

    def __init__(self, alpha=0.1, alpha_calc=None, iter_type='ART'):
        super().__init__(alpha, alpha_calc)
        self.shape = None
        self.wi = None
        if iter_type not in self.iter_types:
            raise ValueError(" Iterator type {} is not supported. Supported iterator types: {}."
                             .format(iter_type, self.iter_types))
        self.iter_type = self.iter_types.index(iter_type)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = np.zeros(shape)
        self.shape = model.solution.shape
        self.wi = np.sum(np.square(model.detector_geometry), axis=tuple(range(1, model.detector_geometry.ndim)))

    def finalize(self, model):
        pass

    def __str__(self):
        return self.iter_types[self.iter_type]

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        # multiplication
        for i in range(model.detector_signal.shape[0]):
            y = signal.get_signal_one_det(model.solution, model.detector_geometry[i])
            dp = model.detector_signal[i] - y
            if self.wi[i] != 0:
                ai = dp / self.wi[i]
            else:
                ai = 0
            if self.iter_type == 1:  # MART
                if model.detector_signal[i] != 0:
                    ai = ai / np.abs(model.detector_signal[i])
            model.solution = model.solution + ai * model.detector_geometry[i] * alpha


class _ARTGPU(abstract_iterator.AbstractIterator):
    iter_types = ('ART', 'MART')

    def __init__(self, alpha=0.1, alpha_calc=None, iter_type='ART'):
        super().__init__(alpha, alpha_calc)
        self.shape = None
        self.wi = None
        if iter_type not in self.iter_types:
            raise ValueError(" Iterator type {} is not supported. Supported iterator types: {}."
                             .format(iter_type, self.iter_types))
        self.iter_type = self.iter_types.index(iter_type)

    def init(self, model, steps, *args, **kwargs):
        super().init(model, steps, *args, **kwargs)
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = cp.zeros(shape)
        self.shape = model.solution.shape
        model.detector_geometry = cp.array(model.detector_geometry)
        model.solution = cp.array(model.solution)
        model._detector_signal = cp.array(model.detector_signal)
        self.wi = cp.sum(cp.square(model.detector_geometry), axis=tuple(range(1, model.detector_geometry.ndim)))

    def finalize(self, model):
        model.detector_signal = cp.asnumpy(model.detector_signal)
        model.detector_geometry = cp.asnumpy(model.detector_geometry)
        model.solution = cp.asnumpy(model.solution)

    def __str__(self):
        return self.iter_types[self.iter_type] + " GPU version"

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        # multiplication
        for i in range(model.detector_signal.shape[0]):
            y = cp.sum(cp.multiply(model.solution, model.detector_geometry[i]))
            dp = model.detector_signal[i] - y
            if self.wi[i] != 0:
                ai = dp / self.wi[i]
            else:
                ai = 0
            if self.iter_type == 1:  # MART
                if model.detector_signal[i] != 0:
                    ai = ai / cp.abs(model.detector_signal[i])
            model._solution = model.solution + ai * model.detector_geometry[i] * alpha



class SIRT(IteratorFactory):
    """A set of iterative algebraic algorithms for image reconstruction.

    see E.F. Oliveira et. al., "Comparison among tomographic reconstruction algorithms with limited data."
    in case of SIRT averaged correction is applied at the end of iteration.
    In the case of several projections in SIRT averaged correction is applied after iteration over each slide

    GPU acceleration is supported. Requires cupy to be installed.
    To turn it on run script with environmental variable TM_GPU set to "1"
    Or just write in your script:
       import os
       os.environ["TM_GPU"] = "1"
    """
    pass


class _SIRTCPU(_ARTCPU):
    iter_types = ('SIRT', 'SMART')

    def __init__(self, alpha=0.1, alpha_calc=None, iter_type='SIRT', n_slices=1):
        super().__init__(alpha, alpha_calc, iter_type)
        self.n_slices = n_slices

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        # multiplication
        det_num = model.detector_signal.shape[0]
        for i in range(self.n_slices):
            # get slice
            # ############## may be speeded up if all slices are calc. once out of for cycle
            i1 = int(i * np.ceil(det_num / self.n_slices))
            i2 = int(min((i + 1) * np.ceil(det_num / self.n_slices), det_num))
            w_slice = model.detector_geometry[i1:i2]
            wi_slice = self.wi[i1:i2]
            y_slice = model.detector_signal[i1:i2]
            # calculating  correction
            p = signal.get_signal(model.solution, w_slice)
            dp = y_slice - p
            a = np.divide(dp, wi_slice, out=np.zeros_like(dp), where=wi_slice != 0)

            if self.iter_type == 1:  # SMART
                a = np.divide(a, np.abs(y_slice), out=np.zeros_like(a), where=y_slice > 1E-20)
            correction = alpha / (i2 - i1) * np.sum(np.multiply(np.moveaxis(w_slice, 0, -1), a), axis=-1)
            model.solution = model.solution + correction


class _SIRTGPU(_ARTGPU):
    iter_types = ('SIRT', 'SMART')

    def __init__(self, alpha=0.1, alpha_calc=None, iter_type='SIRT', n_slices=1):
        super().__init__(alpha, alpha_calc, iter_type)
        self.n_slices = n_slices

    def step(self, model, step_num):
        alpha = self.get_alpha(model, step_num)
        # multiplication
        det_num = model.detector_signal.shape[0]
        for i in range(self.n_slices):
            # get slice
            # ############## may be speeded up if all slices are calc. once out of for cycle
            i1 = int(i * cp.ceil(det_num / self.n_slices))
            i2 = int(min((i + 1) * cp.ceil(det_num / self.n_slices), det_num))
            w_slice = model.detector_geometry[i1:i2]
            wi_slice = self.wi[i1:i2]
            y_slice = model.detector_signal[i1:i2]
            # calculating  correction
            p = signal.get_signal_gpu(model.solution, w_slice)
            dp = y_slice - p
            a = cp.divide(dp, wi_slice)
            a = cp.where(cp.isnan(a), 0, a)

            if self.iter_type == 1:  # SMART
                y_slice = cp.where(y_slice < 1E-20, 0, y_slice)
                a = cp.divide(a, np.abs(y_slice))
                a = cp.where(cp.isnan(a), 0, a)
            correction = alpha / (i2 - i1) * cp.sum(cp.multiply(cp.moveaxis(w_slice, 0, -1), a), axis=-1)
            model._solution = model.solution + correction


