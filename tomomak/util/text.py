import collections


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

