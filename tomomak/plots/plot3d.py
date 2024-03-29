import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
try:
    from mayavi import mlab
    from traits.api import HasTraits, Range, Instance, on_trait_change, Float
    from traitsui.api import View, Item, HGroup, Group
    from tvtk.pyface.scene_editor import SceneEditor
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from tvtk.util.ctf import ColorTransferFunction
except ImportError:
    mlab = None


def _build_voxel_plot(scene, x, y, z, faces, data, title='', axes=False, colormap='blue-red', limits=None):
    scene.background = (1, 1, 1)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    data = np.array(data)
    faces = np.array(faces)
    if limits is None:
        obj = scene.mlab.triangular_mesh(x, y, z, faces, scalars=data, colormap=colormap)
    else:
        obj = scene.mlab.triangular_mesh(x, y, z, faces, scalars=data, colormap=colormap,
                                         vmin=limits[0], vmax=limits[1])
    cb = scene.mlab.colorbar(title=title, orientation='vertical')
    cb.title_text_property.color = (0, 0, 0)
    cb.label_text_property.line_offset = 5
    cb.label_text_property.color = (0, 0, 0)
    cb.scalar_bar.unconstrained_font_size = True
    cb.title_text_property.font_size = 12
    # axes
    if axes:
        ax = scene.mlab.axes(xlabel=axes[0], ylabel=axes[1], zlabel=axes[2], nb_labels=5, color=(0, 0, 0))
        ax.title_text_property.color = (0, 0, 0)
        ax.label_text_property.color = (0, 0, 0)
    return obj


def voxel_plot(data, x, y, z, faces, title='',  axes=False, colormap='blue-red', limits=None):
    if mlab is None:
        raise ImportError("Unable to import Mayavi for 3D visualization.")

    class Visualization(HasTraits):
        d_min = float(np.amin(data))
        d_max = float(np.amax(data))
        min_val = Range(d_min, d_max, d_min)
        max_val = Range(d_min, d_max, d_max)
        scene = Instance(MlabSceneModel, ())

        def __init__(self):
            self.plot = None
            HasTraits.__init__(self)

        @on_trait_change('scene.activated')
        def setup(self):
            self.plot =  _build_voxel_plot(self.scene, x, y, z, faces, data, title, axes, colormap, limits)

        def _update_plot(self):
            new_faces = []
            min_val = self.min_val
            max_val = self.max_val
            # dealing with traits bug
            if min_val < self.d_min:
                min_val = self.d_min
            if max_val > self.d_max:
                max_val = self.d_max
            # add only needed faces
            for i, f in enumerate(faces):
                if min_val <= data[f[0]] <= max_val:
                    new_faces.append(f)
            if min_val < max_val:
                if new_faces:
                    self.plot.mlab_source.trait_set(triangles=new_faces)

        @on_trait_change('min_val')
        def update_min(self):
            self._update_plot()

        @on_trait_change('max_val')
        def update_max(self):
            self._update_plot()

        # the layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=700, width=800, show_label=False),
                    Group('_', 'min_val', 'max_val'), resizable=True, title="Density")
    visualization = Visualization()
    visualization.configure_traits()


def detector_voxel_plot(data, x, y, z, faces, title='',  axes=False, colormap='blue-red', limits=None):
    if mlab is None:
        raise ImportError("Unable to import Mayavi for 3D visualization.")

    class Visualization(HasTraits):
        d_min = float(np.amin(data))
        d_max = float(np.amax(data))
        min_val = Range(d_min, d_max, d_min)
        max_val = Range(d_min, d_max, d_max)
        Detector = Range(1, len(data), 1)
        scene = Instance(MlabSceneModel, ())

        def __init__(self):
            self.plot = None
            self.index = 0
            HasTraits.__init__(self)

        @on_trait_change('scene.activated')
        def setup(self):
            self.plot = _build_voxel_plot(self.scene, x, y, z, faces, data[0], title, axes, colormap, limits)

        def _update_plot(self):
            new_faces = []
            min_val = self.min_val
            max_val = self.max_val
            # dealing with traits bug
            if min_val < self.d_min:
                min_val = self.d_min
            if max_val > self.d_max:
                max_val = self.d_max
            # add only needed faces
            for i, f in enumerate(faces):
                if min_val <= data[self.index][f[0]] <= max_val:
                    new_faces.append(f)
            if min_val < max_val:
                if new_faces:
                    self.plot.mlab_source.trait_set(triangles=new_faces)

        @on_trait_change('min_val')
        def update_min(self):
            self._update_plot()

        @on_trait_change('max_val')
        def update_max(self):
            self._update_plot()

        @on_trait_change('Detector')
        def update_plot(self):
            self.index = self.Detector - 1
            self.plot.mlab_source.trait_set(scalars=data[self.index])
            self._update_plot()

        # the layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=700, width=800, show_label=False),
                    Group('_', 'min_val', 'max_val', 'Detector'), resizable=True, title="Detectors")

    visualization = Visualization()
    visualization.configure_traits()


def _build_contour3d(data, x, y, z, scene, title='', colormap='blue-red', limits=None, style=3, axes=False):
    """Basic 3d visualization using mayavi contour3d graph.

    The x, y and z arrays are then supposed to have been generated by numpy.mgrid,
    in other words, they are 3D arrays, with positions lying on a 3D orthogonal and regularly spaced grid
    with nearest neighbor in space matching nearest neighbor in the array.

    To get publication-quality picture you will probably need to do fine-tuning in Mayavi UI.

    Args:
        data (3D ndarray): visualized density.
        x (3D ndarray): x grid coordinates.
        y (3D ndarray): y grid coordinates.
        z (3D ndarray): z grid coordinates.
        scene (mayavi scene): mayavi scene to work with.
        title (str, optional): Plot title. Default: ''.
        colormap (str, optional): Colormap. Default: 'blue-red.
        limits (tuple of 2 floats, optional): If not None, this tuple represents (lowest, highest) visible densities.
            If None, limits are chosen automatically. Default: None.
        style (int, optional): predefined plot style. Available options: 1, 2, 3, 33. Default: 3.
        axes (False or tuple of 3 str): If not False, axes are shown. Axes names are represented by tuple values.
            Default: False.

    Returns:
        mayavi object: contour3d object.
    """

    # graph
    scene.background = (1, 1, 1)
    min_ax_size = np.min(x.shape)
    if style == 1 or style == 2:
        obj = scene.mlab.contour3d(x, y, z, data, colormap=colormap, opacity=0.05)
        if style == 1:
            obj.actor.property.line_width = 200/min_ax_size
            obj.actor.property.representation = 'wireframe'
            obj.actor.property.lighting = False
            obj.contour.number_of_contours = 20
        elif style == 2:
            obj.actor.property.representation = 'points'
            obj.actor.property.lighting = False
            obj.actor.property.point_size = 200/min_ax_size
        if limits is None:
            obj.contour.minimum_contour = np.amax(data) * 0.2
            obj.contour.auto_update_range = False
        else:
            obj.contour.minimum_contour = limits[0]
            obj.contour.maximum_contour = limits[1]
    elif style == 3 or style == 33:
        if limits is None:
            max_val = np.amax(data)
            min_val = max_val * 0.2
        else:
            min_val = limits[0]
            max_val = limits[1]

        if style == 33:  # special case to deal with Mayavi bug
            obj = scene.mlab.pipeline.volume(mlab.pipeline.scalar_field(x, y, z, data))
        else:
            obj = scene.mlab.pipeline.volume(mlab.pipeline.scalar_field(x, y, z, data), vmin=min_val, vmax=max_val)
        # alpha channel
        otf = obj._otf
        otf.remove_all_points()
        otf.add_point(min_val, 0)
        otf.add_point(min_val + (max_val - min_val) * 0.3, 0.6)
        otf.add_point(max_val, 0.6)
        obj.volume.property.shade = False
    else:
        raise ValueError("Style {} is not supported".format(style))

    # colorbar
    cb = scene.mlab.colorbar(title=title, orientation='vertical')
    cb.title_text_property.color = (0, 0, 0)
    cb.label_text_property.line_offset = 5
    cb.label_text_property.color = (0, 0, 0)
    cb.scalar_bar.unconstrained_font_size = True
    cb.title_text_property.font_size = 12
    # axes
    if axes:
        ax = scene.mlab.axes(xlabel=axes[0], ylabel=axes[1], zlabel=axes[2], nb_labels=5, color=(0, 0, 0))
        ax.title_text_property.color = (0, 0, 0)
        ax.label_text_property.color = (0, 0, 0)
    return obj


def contour3d(data, x, y, z,  title='', colormap='blue-red', limits=None, style=3, axes=False, faces=None):
    """Basic 3d visualization for Solution.

    The x, y and z arrays are then supposed to have been generated by numpy.mgrid,
    in other words, they are 3D arrays, with positions lying on a 3D orthogonal and regularly spaced grid
    with nearest neighbor in space matching nearest neighbor in the array.

    To get publication-quality picture you will probably need to do fine-tuning in Mayavi UI.

    Args:
        data (3D ndarray): visualized density.
        x (3D ndarray): x grid coordinates.
        y (3D ndarray): y grid coordinates.
        z (3D ndarray): z grid coordinates.
        title (str, optional): Plot title. Default: ''.
        colormap (str, optional): Colormap. Default: 'blue-red.
        limits (tuple of 2 floats, optional): If not None, this tuple represents (lowest, highest) visible densities.
            If None, limits are chosen automatically. Default: None.
        style (int, optional): predefined plot style. Available options: 1, 2, 3, 33. Default: 3.
        axes (False or tuple of 3 str): If not False, axes are shown. Axes names are represented by tuple values.
            Default: False.
    """
    if mlab is None:
        raise ImportError("Unable to import Mayavi for 3D visualization.")

    class Visualization(HasTraits):
        scene = Instance(MlabSceneModel, ())

        def __init__(self):
            self.index = 0
            HasTraits.__init__(self)

        @on_trait_change('scene.activated')
        def setup(self):
            _build_contour3d(data, x, y, z, self.scene, title, colormap, limits, style, axes)

        # the layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                         height=700, width=800, show_label=False), resizable=True, title="Density")

    visualization = Visualization()
    visualization.configure_traits()


def detector_contour3d(data, x, y, z,  title='', colormap='blue-red', limits=None, style=3,
                       axes=False, equal_norm=False):
    """Basic 3d visualization for detectors.

      The x, y and z arrays are then supposed to have been generated by numpy.mgrid,
      in other words, they are 3D arrays, with positions lying on a 3D orthogonal and regularly spaced grid
      with nearest neighbor in space matching nearest neighbor in the array.

      To get publication-quality picture you will probably need to do fine-tuning in Mayavi UI.

      Args:
          data (4D ndarray): 3D data for each detectors.
          x (3D ndarray): x grid coordinates.
          y (3D ndarray): y grid coordinates.
          z (3D ndarray): z grid coordinates.
          title (str, optional): Plot title. Default: ''.
          colormap (str, optional): Colormap. Default: 'blue-red.
          limits (tuple of 2 floats, optional): If not None, this tuple represents (lowest, highest) visible densities.
              If None, limits are chosen automatically. Default: None.
          style (int, optional): predefined plot style. Available options: 1, 2, 3, 33. Default: 3.
          axes (False or tuple of 3 str): If not False, axes are shown. Axes names are represented by tuple values.
              Default: False.
          equal_norm(bool, optional): If True,  all detectors will have same z axis.
            If False, each detector has individual z axis. default:False
      """
    if mlab is None:
        raise ImportError("Unable to import Mayavi for 3D visualization.")
    # Currently there is a bug with Mayavi vmin and vmax limits in interactive volume render.
    # So special rendering case is introduced.
    if style == 3:
        style = 33

    class Visualization(HasTraits):
        Detector = Range(1, data.shape[0], 1)
        scene = Instance(MlabSceneModel, ())

        def __init__(self):
            self.index = 0
            self.plot = None
            HasTraits.__init__(self)

        @on_trait_change('scene.activated')
        def setup(self):

            self.plot = _build_contour3d(data[self.index], x, y, z,
                                         self.scene, title, colormap, limits, style, axes)
            if equal_norm:
                if style != 33:
                    max_d = np.max(data)
                    min_d = np.min(data)
                    self.plot.contour.minimum_contour = np.amax(data) * 0.2
                    self.plot.module_manager.scalar_lut_manager.use_default_range = False
                    self.plot.module_manager.scalar_lut_manager.data_range = [min_d, max_d]

                else:
                    warnings.warn("Equal norm is not supported for this style due to Mayavi bugs.")

        @on_trait_change('Detector')
        def update_plot(self):
            self.index = self.Detector - 1
            self.plot.mlab_source.trait_set(scalars=data[self.index])

        # the layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=700, width=800, show_label=False),
                    HGroup('_', 'Detector'), resizable=True, title="Detectors")

    visualization = Visualization()
    visualization.configure_traits()


def mesh3d(data, axis1,  axis2, axis3,  title='', fill_scheme='viridis', *args, **kwargs):
    (vertex, faces) = axis1.cell_edges3d_cartesian(axis2, axis3)
    norm = matplotlib.colors.Normalize(vmin=np.amin(data), vmax=np.amax(data))
    data = norm(data)
    colormap = plt.get_cmap(fill_scheme)
    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    #
    #           Z
    #           ^
    #           | pt1_ _ _ _ _ _ _ _ _ _ _pt2
    #           |  /|    Y                /|
    #           | / |   ^                / |
    #         pt3/_ | _/_ _ _ _ _ _ _pt4/  |
    #           |   | /                |   |
    #           |   |/                 |   |
    #           |  pt5_ _ _ _ _ _ _ _ _|_ _|pt6
    #           |  /                   |  /
    #           | /                    | /
    #        pt7|/_ _ _ _ _ _ _ _ _ _ _|/pt8_______\X
    #                                              /
    mlab.figure(figure="Solution", bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(400, 350))
    for i, (v2, d2) in enumerate (zip(vertex, data)):
        for j, (v1, d1) in enumerate(zip(v2, d2)):
            for k, (v, d) in enumerate(zip(v1, d1)):
                # v_ar = np.array(v)
                # print(v_ar.shape)
                # v_ar = np.swapaxes(v_ar, 0, 1)
                # print(v_ar)
                # mlab.mesh(v_ar[0], v_ar[1], v_ar[2], colormap="bone")
                if d > 0.5:
                    x1, y1, z1 = v[6]  # | => pt1 (0, 1, 1)
                    x2, y2, z2 = v[4]  # | => pt2 (1, 1, 1)
                    x3, y3, z3 = v[7]  # | => pt3 (0, 0, 1)
                    x4, y4, z4 = v[3]  # | => pt4 (1, 0, 1)
                    x5, y5, z5 = v[5]  # | => pt5 (0, 1, 0)
                    x6, y6, z6 = v[2]  # | => pt6 (1, 1, 0)
                    x7, y7, z7 = v[0]  # | => pt7 (0, 0, 0)
                    x8, y8, z8 = v[1]  # | => pt8 (1, 0, 0)
                    color = colormap(d)[0:3]
                    m_color = color
                    mlab.mesh([[x1, x2],
                                      [x3, x4]],  # | => x coordinate

                                     [[y1, y2],
                                      [y3, y4]],  # | => y coordinate

                                     [[z1, z2],
                                      [z3, z4]],  # | => z coordinate
                                     color=m_color,
                                     )  # wireframe or surface

                    mlab.mesh([[x5, x6], [x7, x8]],
                                         [[y5, y6], [y7, y8]],
                                         [[z5, z6], [z7, z8]],
                                         color=m_color,
                                         )
                    mlab.mesh([[x1, x3], [x5, x7]],
                                         [[y1, y3], [y5, y7]],
                                         [[z1, z3], [z5, z7]],
                                         color=m_color,
                                         )
                    mlab.mesh([[x1, x2], [x5, x6]],
                                         [[y1, y2], [y5, y6]],
                                         [[z1, z2], [z5, z6]],
                                         color=m_color,
                                         )
                    mlab.mesh([[x2, x4], [x6, x8]],
                                         [[y2, y4], [y6, y8]],
                                         [[z2, z4], [z6, z8]],
                                         color=m_color,
                                        )

                    mlab.mesh([[x3, x4], [x7, x8]],
                                         [[y3, y4], [y7, y8]],
                                         [[z3, z4], [z7, z8]],
                                         color=m_color,
                                         )

    mlab.show()
