import os
import importlib
from decorator import decorator
import multiprocessing
import warnings


class IteratorFactory:
    """Abstract factory which automatically returns GPU or CPU version of iterator.
    """
    def __new__(cls, *args, **kwargs):
        gpu_enable = os.getenv('TM_GPU')
        if gpu_enable is not None and gpu_enable != 0 and gpu_enable != '0' and gpu_enable.lower != 'false':
            try:
                import cupy as cp
            except ImportError:
                try:
                    cp = importlib.util.find_spec("cupy")
                    raise ImportError("Unable to import CuPy for GPU-acceleration. Try to reinstall it.")
                except ImportError:
                    raise ImportError("Unable to import CuPy for GPU-acceleration. CuPy is not installed")
            module = importlib.import_module(cls.__module__)
            class_name = '_' + cls.__name__ + 'GPU'
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)
        else:
            module = importlib.import_module(cls.__module__)
            class_name = '_' + cls.__name__ + 'CPU'
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)


@decorator
def muti_proc(func, *args, **kwargs):
    mp_enable = os.getenv('TM_MP')
    name = func.__name__
    module = importlib.import_module(func.__module__)
    if mp_enable is not None and mp_enable != 0 and mp_enable != '0' and mp_enable.lower != 'false':
        if int(mp_enable) > multiprocessing.cpu_count():
            warnings.warn("Number of desired cores, passed through TM_MP environmental variable "
                          " > number of available cpu cores.")
        if int(mp_enable) < 2:
            warnings.warn("Number of desired cores, passed through TM_MP environmental variable is < 2."
                          " Multiprocessing will not have any effect.")
        new_name = '_' + name + '_mp'
    else:
        new_name = '_' + name
    new_func = getattr(module, new_name)
    return new_func(*args, **kwargs)

