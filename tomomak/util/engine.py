import os
import importlib


class IteratorFactory:
    """Abstract factory which automatically returns GPU or CPU version of iterator.
    """
    def __new__(cls, *args, **kwargs):
        gpu_enable = os.getenv('TM_GPU')
        if gpu_enable != 0 and gpu_enable != '0' and gpu_enable.lower != 'false':
            module = importlib.import_module(cls.__module__)
            class_name = '_' + cls.__name__ + 'GPU'
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)
        else:
            module = importlib.import_module(cls.__module__)
            class_name = '_' + cls.__name__ + 'CPU'
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)
