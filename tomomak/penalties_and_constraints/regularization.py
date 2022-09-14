from ..iterators import abstract_iterator
import numpy as np
from tomomak.util.engine import IteratorFactory

class Tikhonov(IteratorFactory):
    """Extended version of Tikhonov regularization vs selectable norm.

    Standard cases:
    Norm = 1 - Tikhonov (Ridge).
    Norm = 2 - Lasso.


    """
    pass


class _PositiveCPU(abstract_iterator.AbstractIterator):

    def __init__(self):
        super().__init__()
        pass
