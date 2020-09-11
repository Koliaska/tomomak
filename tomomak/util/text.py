import collections
import time
import tomomak.mesh.cartesian


def density_units(units):
    """Combine axes units  to get density units.

    Args:
        units (iterable of str): list of each axis units.

    Returns:
        str: density units

    """
    counter = collections.Counter(units)
    res = ''
    for k in counter.keys():
        res += '{}{}'.format(k, ('$^{-' + str(counter[k]) + '}$'))
    return res


def detector_caption(mesh, data_type, cartesian=False):
    """Return caption for detector intersection plots.

    Args:
        mesh (tomomak mesh): mesh for units extraction.
        data_type (str): type of the data: 'detector_geometry' or 'detector_geometry_n'.
        cartesian (bool, optional): If True, volumes in cartesian coordinates. Default: False.

    Returns:
        str: plot caption
    """
    if data_type not in ['detector_geometry', 'detector_geometry_n']:
        raise ValueError('Incorrect data type: correct types are: detector_geometry or detector_geometry_n')
    cart_units = None
    if cartesian:
        for a in mesh.axes:
            if type(a) is tomomak.mesh.cartesian.Axis1d and a.spatial:
                cart_units = a.units
    if data_type == 'detector_geometry':
        units = [a.units for a in mesh.axes]
        for i, _ in enumerate(units):
            if mesh.axes[i].spatial:
                if cart_units is not None and cartesian:
                    units[i] = cart_units
        units = ', ' + density_units(units).replace('-', '')
    else:
        units = ''
    if len(mesh.axes) == 1:
        vol_name = 'Length'
    elif len(mesh.axes) == 2:
        vol_name = 'Area'
    else:
        vol_name = 'Volume'
    if data_type == 'detector_geometry_n':
        vol_name = 'Normalized ' + vol_name
    title = r"{}{}".format(vol_name, units)
    return title


def solution_caption(cartesian, *axes):
    cart_units = None
    if cartesian:
        for a in axes:
            if type(a) is tomomak.mesh.cartesian.Axis1d and a.spatial:
                cart_units = a.units
    units = [a.units for a in axes]
    for i, u in enumerate(units):
        if axes[i].spatial:
            if cart_units is not None and cartesian:
                units[i] = cart_units
    units = density_units(units)
    title = r"Density, {}".format(units)
    return title


def progress_mp(res, task_num):
    is_run = True
    while is_run:
        running, successful = 0, 0
        for result in res:
            try:
                if result.successful():
                    successful += 1
            except ValueError:
                running += 1
        print('\r', end='')
        print("Generating array of fan detectors: ", str(successful * 100 // task_num) + "% complete", end='')
        time.sleep(1)
        if successful == task_num:
            is_run = False
    print('\r \r', end='')
    print('\r \r', end='')
