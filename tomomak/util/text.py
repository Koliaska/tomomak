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

