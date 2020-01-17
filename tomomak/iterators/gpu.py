from . import abstract_iterator
import numpy as np
import warnings
import cupy as cp


class MLCuda(abstract_iterator.AbstractIterator):
    """Maximum likelihood iterative solver for image reconstruction using GPU for calculation.
      see  tomomak.iterators.ml.ML for description.
      Warning: GPU calculations are experimental.
      Requires cupy to be installed.
      If you want to use statistics or constraints, GPU version of these objects are needed.
      Alpha should be cupy array.
      Typical performance improvement is 6x for GTX 1060 or 50x for TITAN as compared to core i7-7700.
      """

    def __init__(self):
        super().__init__(None, None)
        self.w_det = None
        self.wi = None
        self.shape = None
        self.y_expected = None
        self.y_len = None
        self.mult = None

    def init(self, model, steps, *args, **kwargs):
        # super().init(model, steps, *args, **kwargs)
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = cp.ones(shape)
        else:
            if cp.all(model.solution):
                warnings.warn("Some elements in model solution are zero. They will not be changed.")
        warnings.warn("Cuda ML is an experimental feature.")
        self.shape = model.solution.shape
        self.w_det = cp.array(np.multiply(np.moveaxis(model.detector_geometry, 0, -1), model.detector_signal))
        self.wi = cp.array(np.sum(model.detector_geometry, axis=0))
        model.detector_geometry = cp.array(model.detector_geometry)
        model.solution = cp.array(model.solution)
        self.y_expected = cp.zeros(model.detector_signal.shape)
        self.y_len = self.y_expected.shape[0]
        self.mult = cp.array(self.w_det)

    def finalize(self, model):
        model.detector_geometry = cp.asnumpy(model.detector_geometry)
        model.solution = cp.asnumpy(model.solution)

    def __str__(self):
        return 'Maximum Likelihood method (GPU version)'

    def step(self, model, step_num):
        # expected signal
        for i in range(self.y_len):
            tmp = cp.multiply(model.solution, model.detector_geometry[i])
            self.y_expected[i] = cp.sum(tmp)
        # multiplication
        self.mult = cp.divide(self.w_det, self.y_expected)
        self.mult = cp.where(cp.isnan(self.mult), 0, self.mult)
        self.mult = cp.sum(self.mult, axis=-1)
        self.mult /= self.wi
        # find delta
        model.solution = model.solution * self.mult


