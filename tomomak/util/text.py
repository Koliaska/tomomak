import collections
import time


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


def detector_caption(mesh):
    """Return caption for detector intersection plots.

    Args:
        mesh (tomomak mesh): mesh for units extraction.

    Returns:
        str: plot caption
    """
    units = density_units([a.units for a in mesh.axes]).replace('-', '')
    if len(mesh.axes) == 1:
        vol_name = 'Length'
    elif len(mesh.axes) == 2:
        vol_name = 'Surface'
    else:
        vol_name = 'Volume'
    title = r"{}, {}".format(vol_name, units)
    return title


def progress_mp(res, task_num):
    is_run = True
    while is_run:
        running, successful= 0, 0
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
