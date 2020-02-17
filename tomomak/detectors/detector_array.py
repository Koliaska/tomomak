from tomomak.util.engine import muti_proc
from multiprocessing import Pool
import numpy as np
import os
import importlib


@muti_proc
def detector_array(func_name, kwargs_list):
    """Generate array of func_name detectors using list of dicitionaries as parameters.

    Multiprocess acceleration is supported.
    To turn it on run script with environmental variable TM_MP set to number of desired cores.
    Or just write in your script:
       import os
       os.environ["TM_MP"] = "8"
    If you use Windows, due to Python limitations, you have to guard your script with
    if __name__ == "__main__":
        ...your script

    Args:
        func_name (string): function name. Required format is "module_name.function_name".
        kwargs_list (list of dictionaries): list of arguments.
        For each element of the list target function will be executed,
        using key-value pairs in this element as **kwargs.

    Returns:
        ndarray: numpy array, representing array of detectors.

    Examples:

        axes = [cartesian.Axis1d(name="X", units="cm", size=50, upper_limit=10),
                cartesian.Axis1d(name="Y", units="cm", size=50, upper_limit=10)]
        m = mesh.Mesh(axes)
        mod = model.Model(mesh=m)
        from tomomak.detectors import detector_array
        kw_list = [dict(mesh=m, p1=(-5, 0), p2=(15, 15), width=0.5, divergence=0.1),
                   dict(mesh=m, p1=(-5, 5), p2=(15, 5), width=0.5, divergence=0.1)]
        det = detector_array.detector_array(func_name='tomomak.detectors.detectors2d.detector2d', kwargs_list=kw_list)
        mod.detector_signal = [0, 0]
        mod.detector_geometry = det
    """
    pass


def _detector_array(func_name, kwargs_list):
    res = []
    module_name, func_name = func_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    list_len = len(kwargs_list)
    for i, k in enumerate(kwargs_list):
        res.append(func(**k))
        print('\r', end='')
        print("Generating detector array of: " + func_name +  str(i*100 // list_len) + " % complete", end='')
    print('\r \r ', end='')
    print('\r \r ', end='')
    return np.array(res)

def _detector_array_mp(func_name, kwargs_list):
    res = []
    module_name, func_name = func_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    proc_num = int(os.getenv('TM_MP'))
    pool = Pool(processes=proc_num)
    print("Started multi-process calculation of {} detector array on {} cores.".format(func_name, proc_num))
    for kw in kwargs_list:
        res.append(pool.apply_async(func, kwds=kw))
    pool.close()
    pool.join()
    final_res = []
    for r in res:
        final_res.append(r.get())
    return np.array(final_res)

