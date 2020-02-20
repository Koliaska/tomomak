import numpy as np
from scipy import interpolate
try:
    import cupy as cp
except ImportError:
    cp = None


def get_signal(solution, detector_geometry):
    """Get detector signals from known object and geometry.

    To find out about solution and detector_geometry see tomomak.model description.

    Args:
        solution (ndarray): known solution.
        detector_geometry (ndarray): known detector geometry.

    Returns:
        ndarray: calculated signals.

    """
    all_but_first = tuple(range(1, solution.ndim + 1))
    signal = np.sum(np.multiply(detector_geometry, solution), axis=all_but_first)
    return signal


def get_signal_gpu(solution, detector_geometry):
    """Get detector signals from known object and geometry. GPU version.

    To find out about solution and detector_geometry see tomomak.model description.

    Args:
        solution (cupy array): known solution.
        detector_geometry (cupy array): known detector geometry.

    Returns:
        ndarray: calculated signals.

    """
    all_but_first = tuple(range(1, solution.ndim + 1))
    signal = cp.sum(cp.multiply(detector_geometry, solution), axis=all_but_first)
    return signal


def get_signal_one_det(solution, one_detector_geometry):
    """Get detector signals from known object and geometry for one detector.

    To find out about solution and detector_geometry see tomomak.model description.

    Args:
        solution (ndarray): known solution.
        one_detector_geometry (ndarray): known detector geometry.

    Returns:
        float: calculated signal
    """
    return np.sum(one_detector_geometry * solution)


def add_noise(signal, st_div):
    """Add gaussian noise to signal.

    Args:
        signal (float): original signal.
        st_div (float): Standard deviation in percent.

    Returns:
        ndarray: numpy array of signal with noise.
    """
    st_div = st_div / 100
    for x in np.nditer(signal, op_flags=['readwrite']):
        x[...] = np.random.normal(x, st_div * x)
    return signal


def resample(signal, rate, n_slices=1):
    """Resample signal adding (rate-1) additional points between each point.

    Args:
        signal (ndarray):
        rate (integer): Resample rate.
        n_slices (integer): number of slices, which should be interpolated  independently, in y
            (e.g. one fan of detectors  = one slice). Slices should have equal number of elements. Default: 1.

    Returns:
        ndarray: 1d array with additional points.
    """
    rate = round(rate)
    det_num = signal.shape[0]
    if det_num % n_slices:
        raise ValueError('Slices are not equal')
    det_num = round(det_num / n_slices)
    y_new = np.zeros(0)
    for i in range(n_slices):
        x = np.arange(det_num)
        f = interpolate.interp1d(x, signal[i * det_num:(i + 1) * det_num], kind='cubic')
        x_new = np.arange((det_num-1) * (rate - 1) + det_num) / rate
        x_new = x_new / x_new[-1] * (det_num-1)
        y_new = np.append(y_new, f(x_new))
    return y_new
