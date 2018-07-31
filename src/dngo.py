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

from .bayesian_linear_regression import BayesianLinearRegression
from .neural_network import zero_mean_unit_var_unnormalization

class BOModel(object):
    # def __enter__(self)
    # def __exit__(self, exc_type, exc_value, tracepck)

    def __init__(self, nn, num_mcmc=0, regressor=None):
        # NN
        self.sess = tf.Session()
        self.nn_model = nn

        # GP
        if regressor is None:
            # self.gp = GPyLinearRegression(num_mcmc=0)
            self.gp = BayesianLinearRegression(num_mcmc=0)
        else:
            self.gp = regressor

    def init(self, X, Y, train_nn=True):
        self.X = X
        self.Y = Y

        # NN
        if train_nn:
            self.nn_model.fit(self.sess, self.X, self.Y)
            self.D = self.nn_model.predict_basis(self.sess, self.X)

        self.gp.fit(self.D, self.nn_model.y)

    def add_observations(self, X_new, Y_new):
        # Update data
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])

        # Fit NN and recalculate basis functions
        self.nn_model.fit(self.sess, self.X, self.Y)
        self.D = self.nn_model.predict_basis(self.sess, self.X)

        # Fit BLR
        self.gp.fit(self.D, self.nn_model.y)

    def get_incumbent(self):
        i = np.argmax(self.Y)
        return self.X[i], self.Y[i]

    def acq(self, X, acq):
        """Note prediction is done in normalized space.
        """

        D = self.nn_model.predict_basis(self.sess, X)
        sample_predictions = self.gp.predict_all(D)

        # Average over all sampled hyperparameter predictions
        return np.average(np.array([acq(mean, var) for (mean, var) in sample_predictions]), axis=0)

    def predict_from_basis(self, D, theta=None):
        mean, var = self.gp.predict(D, theta=theta)

        if self.nn_model.y_std is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.nn_model.y_mean, self.nn_model.y_std)
            var = var * self.nn_model.y_std ** 2

        return mean, var

    def predict(self, X):
        D = self.nn_model.predict_basis(self.sess, X)
        return self.predict_from_basis(D)

    def plot_acq(self, X_line, acq):
        plt.plot(X_line, self.acq(X_line, acq), color="red")

    def plot_prediction(self, X_line, Y_line, x_new=None):
        D_line = self.nn_model.predict_basis(self.sess, X_line)

        if self.gp.num_mcmc > 0:
            for theta in self.gp._current_thetas:
                nnmean = self.nn_model.predict(X_line)
                mean, var = self.predict_from_basis(D_line, theta=theta)
                plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
                plt.plot(X_line, nnmean, color="red")
                # plt.plot(X_line, mean)
        else:
            nnmean = self.nn_model.predict(self.sess, X_line)
            mean, var = self.predict_from_basis(D_line)
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)
            plt.plot(X_line, nnmean, color="red")
            

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        # TODO: remember to normalize if normalization is pulled out into BOModel
        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
        # plt.plot(X_line, self.acq(X_line), color='red')


# Move normalization
# move out session
# save / restore tf model


# class BOModel(object):
#     def __init__(self, nn, kernel, num_mcmc=0, regressor=None):
#         self.gp = GPyRegression(kernel, num_mcmc=0)

#     def init(self, X, Y, train_nn=True):
#         self.X = X
#         self.Y = Y

#         self.gp.fit(self.D, self.nn_model.y)

#     def add_observations(self, X_new, Y_new):
#         # Update data
#         self.X = np.concatenate([self.X, X_new])
#         self.Y = np.concatenate([self.Y, Y_new])

#         # Fit BLR
#         self.gp.fit(self.D, self.nn_model.y)

#     def get_incumbent(self):
#         i = np.argmax(self.Y)
#         return self.X[i], self.Y[i]

#     def acq(self, X, acq):
#         """Note prediction is done in normalized space.
#         """

#         D = self.nn_model.predict_basis(self.sess, X)
#         sample_predictions = self.gp.predict_all(D)

#         # Average over all sampled hyperparameter predictions
#         return np.average(np.array([acq(mean, var) for (mean, var) in sample_predictions]), axis=0)

#     def predict(self, X):
#         D = self.nn_model.predict_basis(self.sess, X)
#         return self.predict_from_basis(D)

#     def plot_acq(self, X_line, acq):
#         plt.plot(X_line, self.acq(X_line, acq), color="red")

#     def plot_prediction(self, X_line, Y_line, x_new=None):
#         D_line = self.nn_model.predict_basis(self.sess, X_line)

#         if self.gp.num_mcmc > 0:
#             for theta in self.gp._current_thetas:
#                 mean, var = self.predict_from_basis(D_line, theta=theta)
#                 plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
#                 plt.plot(X_line, mean)
#         else:
#             mean, var = self.predict_from_basis(D_line)
#             plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
#             plt.plot(X_line, mean)
            

#         if x_new is not None:
#             plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

#         # TODO: remember to normalize if normalization is pulled out into BOModel
#         plt.scatter(self.X, self.Y)
#         plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
#         # plt.plot(X_line, self.acq(X_line), color='red')

#     def __del__(self):
#         self.sess.close()
