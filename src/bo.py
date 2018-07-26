import numpy as np
import tensorflow as tf
import GPy
from GPy.core.parameterization.priors import Gaussian, LogGaussian, Prior
from paramz.domains import _REAL

GPy.plotting.change_plotting_library('matplotlib')

import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy import optimize
from scipy.stats import multivariate_normal

from .dngo import *
from .bayesian_linear_regression import *
from .neural_network import *
from .priors import *
from .acquisition_functions import UCB, EI

def random_grid_samples(n_samples, bounds):
    """Random sample from d-dimensional hypercube (d = bounds.shape[0]).

    Returns: (n_samples, dim)
    """

    dims = bounds.shape[0]
    a = np.random.uniform(0, 1, (dims, n_samples))
    bounds_repeated = np.repeat(bounds[:, :, None], n_samples, axis=2)
    samples = a * np.abs(bounds_repeated[:,1] - bounds_repeated[:,0]) + bounds_repeated[:,0]
    return np.swapaxes(samples, 0, 1)

class BO(object):
    def __init__(self, obj_func, model, acquisition_function=None, n_iter = 10, bounds=np.array([[0,1]])):
        self.n_iter = n_iter
        self.bounds = bounds
        self.obj_func = obj_func
        self.model = model

        if acquisition_function is None:
            self.acquisition_function = UCB(self.model)
        else:
            self.acquisition_function = acquisition_function

    def max_acq(self, n_starts=100):        
        min_y = float("inf")
        min_x = None

        def min_obj(x):
            """lift into array and negate.
            """
            x = np.array([x])
            return -self.model.acq(x, self.acquisition_function.calc)[0]

        # TODO: turn into numpy operations to parallelize
        for x0 in random_grid_samples(n_starts, self.bounds):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 

        return min_x

    def plot_prediction(self, x_new=None):
        X_line = np.linspace(self.bounds[:, 0], self.bounds[:, 1], 100)[:, None]
        Y_line = self.obj_func(X_line)

        plt.subplot(1, 2, 1)
        self.model.plot_prediction(X_line, Y_line, x_new=x_new)
        plt.subplot(1, 2, 2)
        self.model.plot_acq(X_line, self.acquisition_function.calc)
        plt.show()

    def run(self, n_kickstart=2, do_plot=True):
        # Data
        X = random_grid_samples(n_kickstart, self.bounds)
        Y = self.obj_func(X)
        self.model.init(X,Y)

        for i in range(0, self.n_iter):
            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            Y_new = self.obj_func(X_new)

            if do_plot:
                self.plot_prediction(x_new=x_new)

            self.model.add_observations(X_new, Y_new)

        if do_plot:
            self.plot_prediction()
