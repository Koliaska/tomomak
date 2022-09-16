from tomomak import model
from tomomak import solver
from tomomak.test_objects import objects2d
from tomomak.mesh import mesh
from tomomak.mesh.cartesian import Axis1d
import numpy as np
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LogNorm


# This example shows one of the basic solutions for limited data reconstruction: smoothing.
# In order to find best smoothing parameters grid search is used.

# Let's create 2D Cartesian mesh 10 cm x 10 cm which will store our data in  30x30 ndarray.
print("-----------------------------------------------------------------------------------")
print("Ideal case:")
print("-----------------------------------------------------------------------------------")
axes = [Axis1d(name="X", units="cm", size=30, upper_limit=10),
        Axis1d(name="Y", units="cm", size=30, upper_limit=10)]
mesh = mesh.Mesh(axes)

# Now we need some simple object to work with. Let's create circle r = 3 cm in the center of our mesh.
real_solution = objects2d.ellipse(mesh, center=(5, 5), ax_len=(3, 3))
# An easy way to visualize this object is using plot2d() function of the Model class.
# To do this let's create our model and  assign real_solution to it's solution.
mod = model.Model(mesh=mesh, solution=real_solution)
mod.plot2d(fig_name='Test object')
# You should see a discrete circle. Now let's pretend that solution is unknown.
mod.solution = None

# We will start with the ideal case: no signal noise and number of detectors > number of cells.
# Let's create 30 fan detectors around the target. 40 detectors in each fan.
det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=30,
                                     line_num=40,
                                     width=0.5,
                                     divergence=0.05)
mod.detector_geometry = det

# Next step is to simulate each detector signal.
det_signal = signal.get_signal(real_solution, det)
mod.detector_signal = det_signal

# Now let's create solver which will use ART with step = 0.1 in order to find the solution.
solver = solver.Solver()
solver.iterator = algebraic.ART()
solver.iterator.alpha = 0.1
# When you are using ART, often a good idea is to suggest that all values are positive.
c1 = tomomak.constraints.basic.Positive()
solver.constraints = [c1]

# Finally we want some quantitative parameters in order to evaluate reconstruction results.
# Let's count residual mean square and residual norm.
solver.statistics = [statistics.RMS(), statistics.RN()]
# In order to get RMS real solution should be defined.
solver.real_solution = real_solution

# Ok, it's time to solve. Let's do 1000 iterations and see the results.
steps = 1000
solver.solve(mod, steps)
mod.plot2d(fig_name='ART solution after 1000 iterations')
# There are some artifacts but generally picture looks pretty good.
# Now let's plot the statistics.
solver.plot_statistics(fig_name='Reconstruction statistics')
# You can see that both RMS and RN quickly fall during several first steps
# and than begin to slowly reach zero. !!!reach
# You can check yourself that after more iterations
# for the naked eye calculated solution will look exactly like the real one. Just uncomment two following lines.
# solver.solve(mod, 30000)
# mod.plot2d()

# This was pretty easy and pretty unrealistic.
# In the real world applications (especially
# if you work with the limited data tomography)
# you will not meet such conditions.
# Let's consider our first case: limited data. In this example we will use only 10 fans with 30 detectors,
# while mesh is still consist of 30x30 cells.
# Note that we increased the divergence, so detectors cover the entire space.
print("\n-----------------------------------------------------------------------------------")
print("Limited data case (300 detectors, but 900 cells):")
print("-----------------------------------------------------------------------------------")
limited_det = detectors2d.fan_detector_array(mesh=mesh,
                                             focus_point=(5, 5),
                                             radius=11,
                                             fan_num=10,
                                             line_num=30,
                                             width=0.5,
                                             divergence=0.1)

mod.detector_signal = None
mod.detector_geometry = limited_det
mod.detector_signal = signal.get_signal(real_solution, limited_det)

# Now we repeat previous steps and see the results.
solver.refresh_statistics()
mod.solution = None
solver.solve(mod, steps)
mod.plot2d(fig_name='Limited data reconstruction')
solver.plot_statistics(fig_name='Limited data reconstruction statistics')
# The image became more noisy. Also you can notice, that RMS and RN no longer reaching zero.
# They are limited by some value. That is the price you pay for the low number of the detectors.

# Now let's consider high measurements error. For example, this may happen when your detectors are poorly calibrated,
# detector signal has statistical nature, or you don't know exact geometry of the experiment.
# We go back to our 40x30 detectors case but with some Gaussian noises.
print("\n-----------------------------------------------------------------------------------")
print("Noisy signal case:")
print("-----------------------------------------------------------------------------------")
mod.detector_signal = None
mod.detector_geometry = det
noisy_det_signal = signal.add_noise(det_signal, 5)
mod.detector_signal = noisy_det_signal

# We will do the reconstruction in several steps, and see what happens.
solver.refresh_statistics()
mod.solution = None
steps = 20
for i in range(5):
    solver.solve(mod, steps)
    mod.plot2d(fig_name=f'Noisy object after {(i + 1) * steps} steps')
# The first image you see is pretty noisy, but it's only a part of the problem:
# unlike the previous cases, our solution become worse, when we increase the number of steps.

# Let's see the statistics.
solver.plot_statistics(fig_name='Noisy data reconstruction statistics')
# You can see, that RN is decreasing at every step since it is what our iterator doing.
# RMS, however, at some point starts to increase.

# A possible solution to this problem is to use early stopping criteria. This will be discussed in other tutorial.
# Here we will use another trick - additional constraint.
# It is possible to apply 1D function along specific axis at each reconstruction step.
# We will apply scipy.ndimage.gaussian_filter1d, where sigma argument is standard deviation for Gaussian kernel.
func = scipy.ndimage.gaussian_filter1d
c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=0.1, sigma=1)
c3 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=0.1, sigma=1)
solver.constraints = [c1, c2, c3]
# Now we are ready to reconstruct
mod.solution = None
solver.refresh_statistics()
steps = 100
solver.solve(mod, steps)
mod.plot2d(fig_name='Noisy object with early stopping')
solver.plot_statistics(fig_name='Noisy data reconstruction statistics with early stopping')
# Well, the result is clearly better. More importantly our solution converges now.

# The question is: what gaussian_filter1d parameters should we use?
# In order to answer this question we should do hyperparameter optimization.
# Let's implement grid search method which suggest trying many different parameter combinations.
#  We will try some alphas and sigmas and store RMS after 100 steps for each combination. This will take some time.
print("\n-----------------------------------------------------------------------------------")
print("Hyperparameter optimization:")
print("-----------------------------------------------------------------------------------")
res = ""
dat = []
mod.solution = None
solver.refresh_statistics()
steps = 100
for alpha in np.linspace(0.01, 0.3, 20):
    for sig in np.linspace(0.1, 2, 20):
        mod.solution = None
        solver.refresh_statistics()
        c_scan1 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=alpha, sigma=sig)
        c_scan2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=alpha, sigma=sig)
        solver.constraints = [c1, c_scan1, c_scan2]
        solver.solve(mod, steps=steps)
        res += "alpha = {}, sigma = {}, rms = {}\n".format(alpha, sig, solver.statistics[0].data[-1])
        new_dat = [alpha, sig, solver.statistics[0].data[-1]]
        dat.append(new_dat)

# Let's see the graph.
x_list = np.array(dat)[:, 0]
y_list = np.array(dat)[:, 1]
z_list = np.array(dat)[:, 2]
N = int(len(z_list)**.5)
z = z_list.reshape(N, N)
plt.imshow(z, extent=(np.amin(x_list), np.amax(x_list), np.amin(y_list), np.amax(y_list)),
           norm=LogNorm(), aspect='auto')
plt.title("RMS(alpha, sigma)")
plt.xlabel("alpha")
plt.ylabel("sigma")
plt.colorbar()
plt.show()

# There is clear region with low estimated RMS. Let's find a minimum.
min_ind = np.argmin(np.array(dat)[:, 2])
alpha = dat[min_ind][0]
sigma = dat[min_ind][1]
print("\nFinished hyperparameter optimization.")
print(f"Best parameters: alpha = {alpha}, sigma = {sigma}, RMS = {dat[min_ind][2]}")
func = scipy.ndimage.gaussian_filter1d
c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=alpha, sigma=sigma)
c3 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=alpha, sigma=sigma)
solver.constraints = [c1, c2, c3]
mod.solution = None
solver.refresh_statistics()
steps = 100
solver.solve(mod, steps)
mod.plot2d(fig_name='Noisy data object with hyperparameter optimization')
solver.plot_statistics(fig_name='Noisy data reconstruction statistics with optimization')

# This is much better. However don't forget, that parameters we choose
# are the best parameter only for our specific synthetic case.
# If you apply new noise, you will see, that it is true. Good news, that this parameters
# are likely to give you not the best, but almost the best result.

# In this tutorial you've learnt about limited data reconstruction.
# Saw how noisy detector signals influence reconstruction results.
# And tried some tricks to deal with this problem.
