import numpy as np


def multiply_along_axis(a, b, axis):
    """ Multiply numpy array by 1-D numpy array along given axis.

    Args:
        a (ndarray): Array to multiply.
        b (ndarray): 1D  array to multiply by.
        axis (int): Axis along which multiplication is performed.

    Returns:
        ndarray: Multiplied array
    """
    dim_array = np.ones((1, a.ndim), int).ravel()
    dim_array[axis] = -1
    b_reshaped = b.reshape(dim_array)
    return a * b_reshaped


def broadcast_object(ar, index, shape):
    """Broadcast array to a new shape, using given indexes.

    Args:
        ar (ndarray): numpy array to broadcast.
        index (int or tuple of ints):  In the new array input array will be represented using this index or indexes.
        shape (tuple of ints): new  shape.

    Returns:
        ndarray: broadkasted array

    """
    if isinstance(index, int):
        index = [index]
    if ar.shape == shape:
        return ar
    elif len(ar.shape) >= len(shape):
        raise ValueError("New shape should have more dimensions than old shape. "
                         "New shape is {}, old shape is {}".format(ar.shape, shape))
    else:
        # Moving current axes to the end in order to prepare shape for numpy broadcast
        shape = list(shape)
        index = list(index)
        current_shape = []
        for i, ind in enumerate(index):
            val = shape[ind]
            current_shape.append(val)
        sorted_index = np.sort(index)[::-1]
        for i, ind in enumerate(sorted_index):
            del shape[ind]
        for ind in current_shape:
            shape.append(ind)
        # broadcasting
        ar = np.broadcast_to(ar, shape)
        # making correct axes order
        ind_len = len(index)
        ar = np.moveaxis(ar, range(-ind_len, -ind_len + len(index)), index)
        return ar


def normalize_broadcasted(data, index, mesh, data_type):
    full_index = list(range(len(mesh.axes)))
    other_axes = [item for item in full_index if item not in index]
    for i in other_axes:
        dv = mesh.axes[i].volumes
        vol = mesh.axes[i].total_volume
        if data_type == 'solution':
            data = data / vol
        elif data_type == 'detector_geometry':
            data = multiply_along_axis(data, dv, i)

        else:
            raise ValueError("Wrong data type")
    return data




