import matplotlib.pyplot as plt

import tomomak.util.geometry3d_trimesh
from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak import iterators
from tomomak.iterators import ml, algebraic#, gpu
from tomomak.iterators import statistics
import tomomak.constraints.basic
from mpl_toolkits.mplot3d import Axes3D
import itertools


from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import time

#Anaconda3-5.3.1-Windows-x86_64.exe
#conda upgrade -n base -c defaults --override-channels conda


# conda remove vtk conda remove mayavi and installing with pip install vtk and pip install mayavi. Thanks for your help!




from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d, objects3d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
from tomomak.util import geometry3d_basic
import os
import tomomak.constraints.basic
import trimesh
import tomomak.detectors.detectors3d as detectors3d
from scipy.spatial.transform import Rotation as R

from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d
from tomomak.mesh import mesh
from tomomak.mesh import cartesian, polar, toroidal, level
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import numpy as np
import tomomak.util.eqdsk as eqdsk
from tomomak import util
#from mayavi import mlab

import inspect
# import pyvista as pv
from mayavi import mlab


g = eqdsk.read_eqdsk('gglobus32994.g', b_ccw=-1)
eqdsk.calc_rho(g)
g["masked_rho"] = geometry2d.in_out_mask((g['r'], g['z']), (g["rbdry"], g["zbdry"]), in_value=1, out_value=10) * g["rho"]

axes = [toroidal.Axis1d(radius=0.0, name="theta", units="rad", size=6, upper_limit=np.pi/2*3),
        level.Axis1d(level_map=g['masked_rho'], x=g['r'], y=g['z'], x_axis=g['raxis'], y_axis=g['zaxis'], bry_level=0.999, last_level_coordinates=(g["rbdry"], g["zbdry"]),
                     name="rho", units="a.u.",cart_units='m', edges=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,   0.999]),
        polar.Axis1d(name="theta", units="rad", size=20)]
axes[2].RESOLUTION2D = 20
# axes = [polar.Axis1d(name="phi", units="rad", size=12),
#         cartesian.Axis1d(name="R", units="cm", size=15, upper_limit=10)]
m = mesh.Mesh(axes)
mod = model.Model(mesh=m)
res = objects2d.ellipse(m, ax_len=(2, 2), index=(1,2),center=(0.36,0), broadcast=True)
# res= tomomak.util.array_routines.broadcast_object(real_solution, (1,2), m.shape)
# res = tomomak.util.array_routines.normalize_broadcasted(res, (1,2), m, 'solution')
vols = util.geometry3d_trimesh.cell_volumes(m)
areas = util.geometry2d.cell_areas(m, (1, 2))
res = res * vols / areas
#noise = np.random.normal(0, 0.0001, real_solution.shape)
res = util.array_routines.multiply_along_axis(res,np.abs(np.sin(np.linspace(0, np.pi, num=axes[2].size))) + 8, axis=2)
res[:,0,:] = res[:,0,:] / (np.abs(np.sin(np.linspace(0, np.pi, num=axes[2].size))) + 10) * 11
#res[:,0,:] = np.full_like(res[:,0,:] , np.mean(res[:,0,:]* vols[:,0,:] / areas[0,:]))
res = util.array_routines.multiply_along_axis(res,np.linspace(22,10, num=axes[1].size), axis=1)



mod.solution = res

#mod.plot2d(style ='colormesh', cartesian_coordinates=True, index=(1,2))
mod.plot3d(cartesian_coordinates=True, axes=True, style=0)