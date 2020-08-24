"""Module, defining transformation pipeline class and abstract transformer.
"""
import warnings


class Pipeline:
    """Pipeline, providing interface for  model transformation.

        Pipeline is a top-level container for all transforms. It is one of the core TOMOMAK structures.
        Pipeline is usually used to perform a number of transformations before calculation of the solution
        and inverse transformation after solution is obtained.

        Object can be used as a transformer if it implements __call()__ method.
        Model should be changed inside of these methods.
        It should also store information in order to perform inverse transform on the next call.
        Abstract class AbstractTransformer may be used as superclass for transformer.
        See examples.
        """

    def __init__(self, model, transformers=(), names=None):
        self._model = model
        self._position = 0
        self._len = 0
        self._transformers = []
        self._names = []
        for i, t in enumerate(transformers):
            if names is None:
                name = None
            else:
                name = names[i]
            self.add_transform(t, name)

    @property
    def position(self):
        return self._position

    @property
    def transformers(self):
        return self._names

    def add_transform(self, transformer, name=None):
        if name is None:
            name = 'Transformer ' + str(len(self._transformers))
        self._names.append(name)
        self._transformers.append(transformer)
        self._len += 1

    def remove_transform(self, index):
        del self._transformers[index]
        self._len -= 1

    def _check_forward(self, steps, forward=True, type_text='perform transformation'):
        if steps <= 0:
            raise TypeError("Number of steps should be positive.")
        raiser = 0
        if forward:
            if self._position + steps > self._len + 1:
                raiser = 1
        else:
            if self._position - steps < 0:
                raiser = 1
        if raiser:
            raise IndexError("Unable to {} since final index is out of range.".format(type_text))

    def forward(self, steps=1):
        self._check_forward(steps, forward=True, type_text='perform transformation')
        for i in range(steps):
            self._transformers[self._position](self._model)
            self._position += 1

    def backward(self, steps=1):
        self._check_forward(steps, forward=False, type_text='perform transformation')
        for i in range(steps):
            self._transformers[self._position - 1](self._model)
            self._position -= 1

    def to_last(self):
        steps = self._len - self._position
        self.forward(steps)

    def to_first(self):
        steps = self._position
        self.backward(steps)

    def _skip_forward(self, steps=1):
        self._check_forward(steps, forward=True, type_text='skip')
        self._position += steps
        warnings.warn("{} step(s) in the pipeline were skipped forward. "
                      "This may lead to unpredictable results.".format(steps))

    def _skip_backward(self, steps=1):
        self._check_forward(steps, forward=False, type_text='skip')
        self._position -= steps
        warnings.warn("{} step(s) in the pipeline were skipped backward."
                      " This may lead to unpredictable results.".format(steps))
