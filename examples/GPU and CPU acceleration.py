
from tomomak import model
from tomomak.solver import *
from tomomak.test_objects import objects2d
from tomomak import mesh
from tomomak.mesh import cartesian
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml
import os
import time


# This example shows how to accelerate your script using multiprocessing or GPU-acceleration.
# No matter which method you choose, everything is done automatically. You only need to specify desired methods.

# We will start with multiprocessing. Most of the functions and methods are parallelized automatically,
# however some of them need special treatment since they use external libraries.
# This fact is stated in the function docstring.
# To turn parallelization on run script with environmental variable TM_MP set to number of desired cores.
# Or just write in your script:
#        import os
#        os.environ["TM_MP"] = "8"
# If you use Windows, due to Python limitations, you have to guard your script with
# If __name__ == "__main__":

# We will start with If __name__ == "__main__" block
if __name__ == "__main__":

    # Let's create model
    axes = [cartesian.Axis1d(name="X", units="cm", size=40, upper_limit=10),
            cartesian.Axis1d(name="Y", units="cm", size=40, upper_limit=10)]
    mesh = mesh.Mesh(axes)
    mod = model.Model(mesh=mesh)

    # Now let's calculate detector geometry
    start = time.time()
    det = detectors2d.fan_detector_array(mesh=mesh,
                                         focus_point=(5, 5),
                                         radius=11,
                                         fan_num=10,
                                         line_num=40,
                                         width=1,
                                         divergence=0.2)
    end = time.time()
    print("Serial calculation took {} s.".format(end - start))
    # Well, it took some time. Let's visualize acquired geometry.
    mod.detector_geometry = det
    mod.plot2d(data_type='detector_geometry')

    # Now let's activate multiprocessing. We will use 4 cores.
    os.environ["TM_MP"] = "4"
    start = time.time()
    det = detectors2d.fan_detector_array(mesh=mesh,
                                         focus_point=(5, 5),
                                         radius=11,
                                         fan_num=40,
                                         line_num=30,
                                         width=1,
                                         divergence=0.2)
    end = time.time()
    print("Parallel calculation took {} s ".format(end - start))
    # This calculation went faster. Note, that parallelization is effective only for the long calculation.
    # We can visualize new geometry and see that it is the same.
    mod.detector_geometry = det
    mod.plot2d(data_type='detector_geometry')

    # The next step is inverse problem solution. Here you can use GPU acceleration. It requires CuPy to be installed.
    # To turn it on run script with environmental variable TM_GPU set to any nonzero value.
    # Or just write in your script:
    #        import os
    #        os.environ["TM_GPU"] = "1"

    # Let's create synthetic object and find the solution using CPU.
    real_solution = objects2d.ellipse(mesh, (5, 5), (3, 3))
    mod.detector_signal = signal.get_signal(real_solution, det)
    solver = Solver()
    solver.iterator = ml.ML()
    steps = 3000
    solver.solve(mod, steps=steps)

    # Now we let's switch to GPU.
    os.environ["TM_GPU"] = "1"
    mod.solution = None
    # Since you enabled GPU in the middle of the execution, you need to recreate solver.
    # The syntax doesn't change.
    solver.iterator = ml.ML()
    solver.solve(mod, steps=steps)
    # Let's see the result.
    mod.plot2d()
