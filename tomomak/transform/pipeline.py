"""Module, defining transformation pipeline class and abstract transformer.
"""
from abc import ABC, abstractmethod
import warnings


class Pipeline:
    """Pipeline, providing interface for  model transformation.

        Pipeline is a top-level container for all transforms. It is one of the core TOMOMAK structures.
        Pipeline is usually used to perform a number of transformations before calculation of the solution
        and inverse transformation after solution is obtained.

        Object can be used as a transformer if it implements forward(Model) and backward(Model) methods.
        Model should be changed inside of these methods.
        Abstract class AbstractTransformer may be used as superclass for transformer.
        """

    def __init__(self, model, transformers=()):
        self._model = model
        self._position = 0
        self._len = 0
        self._transformers = []
        for t in transformers:
            self._len += 1
            self._transformers.append(t)

    @property
    def position(self):
        return self._position

    def add_transform(self, transformer):
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
            raise Exception("Unable to {} since final index is out of range.".format(type_text))

    def forward(self, steps=1):
        self.__check_forward(steps,forward=True, type_text='perform transformation')
        for i in range(steps):
            self._transformers[self._position].forward(self._model)
            self._position += 1

    def backward(self, steps=1):
        self.__check_forward(steps, forward=False, type_text='perform transformation')
        for i in range(steps):
            self._transformers[self._position - 1].backward(self._model)
            self._position -= 1

    def to_last(self):
        steps = self._len - self._position
        self.forward(steps)

    def to_first(self):
        steps = self._position
        self.backward(steps)

    def skip_forward(self, steps=1):
        self.__check_forward(steps, forward=True, type_text='skip')
        self.position += steps
        warnings.warn("{} steps in the pipeline were skipped forward. "
                      "This may lead to unpredictable results.".format(steps))

    def skip_backward(self, steps=1):
        self.__check_forward(steps, forward=False, type_text='skip')
        self.position -= steps
        warnings.warn("{} steps in the pipeline were skipped backward."
                      " This may lead to unpredictable results.".format(steps))

class AbstractTransformer(ABC):
    """
    
    """

    @abstractmethod
    def forward(self, model):
        """

        """

    @abstractmethod
    def backward(self, model):
        """

        """