import numpy as np
import numbers
import warnings
import copy
import matplotlib.pyplot as plt


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
        # Init iterator and constraints.
        print("Start calculation with {} iterations using {}.".format(steps, self.iterator))
        if self.iterator is not None:
            self.iterator.init(model, steps, *args, **kwargs)
        if self.constraints is not None:
            print("Used constraints:")
            for r in self.constraints:
                r.init(model, steps, *args, **kwargs)
                print("  " + str(r))
        if self.statistics is not None:
            # print("Calculated statistics:")
            for ind, s in enumerate(self.statistics):
                s.init(model, steps, *args, **kwargs)
                # print(" " + str(s))

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

    def plot_statistics(self):
        if self.statistics is not None:
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
