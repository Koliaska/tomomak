from tomomak.util.geometry2d import check_spatial
import pyvista as pv
import numpy as np


def get_grid(mesh, index=(0, 1, 2)):
    """
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    # if mesh.axes[index[0]].dimension == 3:
    #     try:
    #         (vertices, faces) = mesh.axes[index[0]].cell_edges3d_cartesian()
    #         shape = mesh.axes[index[0]].size
    #         pv_list = np.zeros(shape).tolist()
    #         for i, _ in enumerate(pv_list):
    #             if faces is None:
    #                 cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i]))
    #             else:
    #                 cell = trimesh.Trimesh(vertices=vertices[i], faces=faces[i])
    #             pv_list[i] = cell
    #     except (TypeError, AttributeError) as e:
    #         raise type(e)(e.message + "Custom axis should implement cell_edges3d_cartesian method. "
    #                                   " See docstring for more information.")
    # # If 1st axis is 2D
    # elif mesh.axes[index[0]].dimension == 2 or mesh.axes[index[1]].dimension == 2:
    #     (vertices, faces) = mesh.axes_method2d(index, "cell_edges3d_cartesian")
    #     vertices = vertices.tolist()
    #     faces = faces.tolist()
    #     shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size)
    #     pv_list = np.zeros(shape).tolist()
    #     for i, row in enumerate(pv_list):
    #         for j, col in enumerate(row):
    #             if faces is None:
    #                 cell = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=vertices[i][j]))
    #             else:
    #                 cell = trimesh.Trimesh(vertices=vertices[i][j], faces=faces[i][j])
    #             pv_list[i][j] = cell
    #111111111111 change if to elif
    # If axes are 1D
    if mesh.axes[index[0]].dimension == mesh.axes[index[1]].dimension == mesh.axes[index[2]].dimension == 1:
        (vertices, faces) = mesh.axes_method3d(index, "cell_edges3d_cartesian")
        vertices = vertices.tolist()
        faces = faces.tolist()
        shape = (mesh.axes[index[0]].size, mesh.axes[index[1]].size, mesh.axes[index[2]].size)
        pv_list = np.zeros(shape).tolist()
        for i, row in enumerate(pv_list):
            for j, col in enumerate(row):
                for k, _ in enumerate(col):
                    if faces is None:
                        raise TypeError("cell_edges3d_cartesian method of the current axes combination"
                                        " doesn't return faces, which is not supported by pyvista.")
                    else:
                        cell = pv.PolyData(vertices[i][j][k], np.hstack(faces[i][j][k]))
                    pv_list[i][j][k] = cell
    else:
        raise TypeError("3D objects can be built on the 1D,2D and 3D axes only.")

    return pv_list


def show_cell(mesh, index=(0, 1, 2), cell_index=(0, 0, 0)):
    """Shows 3D borders of the given cell.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        index(tuple of one, two or three ints, optional): axes to build object at. Default: (0, 1, 2)
        cell_index(tuple of one, two or three ints, optional): index of the cell. Default: (0, 0, 0)

    Returns:
        None
    """
    if isinstance(index, int):
        index = [index]
    check_spatial(mesh, index)
    pv_list = get_grid(mesh, index)
    for i in cell_index:
        pv_list = pv_list[i]
    pv_list.plot()
