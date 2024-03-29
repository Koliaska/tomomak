import copy
import matplotlib.pyplot as plt
import os
import time
import numpy as np


class Solver:
    """
    Args:

    """
    def __init__(self, iterator=None, constraints=None, statistics=None,
                 stop_condiitons=None, stop_values=None, real_solution=None):
        self.iterator = iterator
        self.constraints = constraints
        self.statistics = statistics
        self.stop_values = stop_values
        self.stop_conditions = stop_condiitons
        self.real_solution = real_solution

    def solve(self, model, steps=20, *args, **kwargs):
        # Check consistency.
        if model.detector_signal is None:
            raise ValueError("detector_signal should be defined to perform reconstruction.")
        if model.detector_geometry is None:
            raise ValueError("detector_geometry should be defined to perform reconstruction.")
        if self.stop_conditions is not None:
            if self.stop_values is None:
                raise ValueError("stop_values should be defined since stop_conditions is defined.")
            if len(self.stop_values) != len(self.stop_conditions):
                raise ValueError("stop_conditions and stop_values have different length.")
        # Check that if GPU-acceleration is enabled, all iterators are capable of GPU-calculation
        gpu_enable = os.getenv('TM_GPU')
        if gpu_enable is not None and gpu_enable != 0 and gpu_enable != '0' and gpu_enable.lower != 'false':
            if self.iterator is not None:
                if type(self.iterator).__name__[-3:] != 'GPU':
                    raise RuntimeError("Iterator doesn't support GPU acceleration")
            if self.constraints is not None:
                for r in self.constraints:
                    if type(r).__name__[-3:] != 'GPU':
                        raise RuntimeError("{} constraint doesn't support GPU acceleration".format(r))
            if self.statistics is not None:
                for ind, s in enumerate(self.statistics):
                    if type(s).__name__[-3:] != 'GPU':
                        raise RuntimeError("{} statistics calculation doesn't support GPU acceleration".format(s))
            if self.stop_conditions is not None:
                print("Stop conditions:")
                for ind, s in enumerate(self.stop_conditions):
                    if type(s).__name__[-3:] != 'GPU':
                        raise RuntimeError("{} stop condition calculation doesn't support GPU acceleration".format(s))
        # Init iterator and constraints.
        print("Started calculation with {} iterations using {}.".format(steps, self.iterator))
        start_time = time.time()
        if self.iterator is not None:
            self.iterator.init(model, steps, *args, **kwargs)
        if self.constraints is not None:
            print("Used constraints:")
            for r in self.constraints:
                r.init(model, steps, *args, **kwargs)
                print("  " + str(r))
        if self.statistics is not None:
            for ind, s in enumerate(self.statistics):
                s.init(model, steps, real_solution=self.real_solution, *args, **kwargs)
                # print(" " + str(s))
        if self.stop_conditions is not None:
            print("Stop conditions:")
            for ind, s in enumerate(self.stop_conditions):
                s.init(model, steps, real_solution=self.real_solution, *args, **kwargs)
                print("{} < {}".format(s, self.stop_values[ind]))

        # Start iteration
        for i in range(steps):
            old_solution = copy.copy(model.solution)
            if self.iterator is not None:
                self.iterator.step(model=model, step_num=i)
            # constraints
            if self.constraints is not None:
                for k, r in enumerate(self.constraints):
                    r.step(model=model, step_num=i)
            # statistics
            if self.statistics is not None:
                for s in self.statistics:
                    s.step(solution=model.solution, step_num=i, real_solution=self.real_solution,
                           old_solution=old_solution, model=model)
            # early stopping
            if self.stop_conditions is not None:
                stop = False
                for k, s in enumerate(self.stop_conditions):
                    val = s.step(solution=model.solution, step_num=i, real_solution=self.real_solution,
                                 old_solution=old_solution, model=model)
                    if val < self.stop_values[k]:
                        print('\r \r', end='')
                        print("Early stopping at step {}: {} < {}.".format(i, s, self.stop_values[k]))
                        stop = True
                if stop:
                    break
            if i % 20 == 0:
                print('\r', end='')
                print("...", str(i * 100 // steps) + "% complete", end='')

        print('\r \r', end='')
        end_time = time.time()
        print(f"Finished in {end_time - start_time} s.")
        if self.iterator is not None:
            self.iterator.finalize(model)
        if self.constraints is not None:
            for r in self.constraints:
                r.finalize(model)
        if self.statistics is not None:
            for s in self.statistics:
                s.finalize(model)
            print("Statistics summary:")
            for s in self.statistics:
                print("  {}: {}".format(s, s.data[-1]))
        model.solution = np.array(model.solution, dtype='float')

    def plot_statistics(self, fig_name='Statistics'):
        if self.statistics is not None:
            plt.figure(fig_name)
            subpl = len(self.statistics) * 100 + 11
            axes = []
            for s in self.statistics:
                ax = plt.subplot(subpl)
                s.plot()
                subpl += 1
                axes.append(ax)
            plt.xlabel('step')
            for ax in axes:
                ax.label_outer()
            # plt.tight_layout()
        else:
            raise Exception("No statistics available.")
        plt.show()
        return axes

    def refresh_statistics(self):
        if self.statistics is not None:
            for s in self.statistics:
                s.data = []
        else:
            raise Exception("No statistics available.")
        print("All collected statistics was deleted.")
