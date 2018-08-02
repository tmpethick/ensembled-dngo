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

from .bayesian_linear_regression import BayesianLinearRegression, GPyRegression
from .neural_network import zero_mean_unit_var_unnormalization
from .normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization

class BOBaseModel(object):
    def get_incumbent(self):
        i = np.argmax(self.Y)
        return self.X[i], self.Y[i]

    def init(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.fit(self.X, self.Y, **kwargs)

    def add_observations(self, X_new, Y_new, **kwargs):
        # Update data
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])
        
        self.fit(self.X, self.Y, **kwargs)

    def plot_acq(self, X_line, acq):
        plt.plot(X_line, self.acq(X_line, acq), color="red")

    def fit(self, X, Y): 
        raise NotImplementedError

    def acq(self, X, acq):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def plot_prediction(self, X_line, Y_line, x_new=None):
        raise NotImplementedError


class BOModel(BOBaseModel):
    # def __enter__(self)
    # def __exit__(self, exc_type, exc_value, tracepck)

    def __init__(self, nn, num_mcmc=0, regressor=None, normalize_input=True, normalize_output=True):
        # NN
        self.sess = tf.Session()
        self.nn_model = nn

        self.X_mean = None
        self.X_std = None
        self.normalize_input = normalize_input

        self.y_mean = None
        self.y_std = None
        self.normalize_output = normalize_output

        # GP
        if regressor is None:
            self.gp = BayesianLinearRegression(num_mcmc=0)
        else:
            self.gp = regressor

    def fit(self, X, Y, train_nn=True):
        if self.normalize_input:
            self.transformed_X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.transformed_X = X

        if self.normalize_output:
            self.transformed_Y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(Y)
        else:
            self.transformed_Y = Y

        # NN
        if train_nn:
            self.nn_model.fit(self.sess, self.transformed_X, self.transformed_Y)
            self.transformed_D = self.nn_model.predict_basis(self.sess, self.transformed_X)

        self.gp.fit(self.transformed_D, self.transformed_Y)

    def acq(self, X, acq):
        """Note prediction is done in normalized space.
        """

        D = self.predict_basis(X)
        # Note: no need to denormalize.
        transformed_sample_predictions = self.gp.predict_all(D)

        asd = np.array([acq(pred[0, :], pred[1, :]) for pred in transformed_sample_predictions])
        # Average over all sampled hyperparameter predictions
        return np.average(asd, axis=0)

    def predict_basis(self, X):
        if self.normalize_input:
            X, _, _ = zero_mean_unit_var_normalization(X, mean=self.X_mean, std=self.X_std)

        return self.nn_model.predict_basis(self.sess, X)
        
    def predict_from_basis(self, transformed_D, theta=None):
        mean, var = self.gp.predict(transformed_D, theta=theta)

        if self.normalize_output is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.y_mean, self.y_std)
            var = var * self.y_std ** 2

        return mean, var

    def predict(self, X):
        D = self.predict_basis(X)
        return self.predict_from_basis(D)

    def plot_prediction(self, X_line, Y_line, x_new=None):
        D_line = self.predict_basis(X_line)

        if self.gp.num_mcmc > 0:
            for theta in self.gp._current_thetas:
                mean, var = self.predict_from_basis(D_line, theta=theta)
                plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
                plt.plot(X_line, mean)
        else:
            mean, var = self.predict_from_basis(D_line)
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')


class GPyBOModel(BOBaseModel):
    def __init__(self, kernel, normalize_input=True, normalize_output=True, **kwargs):
        self.X_mean = None
        self.X_std = None
        self.normalize_input = normalize_input

        self.y_mean = None
        self.y_std = None
        self.normalize_output = normalize_output

        self.gp = GPyRegression(kernel, **kwargs)

    def fit(self, X, Y):
        if self.normalize_input:
            self.transformed_X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.transformed_X = X

        if self.normalize_output:
            self.transformed_Y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(Y)
        else:
            self.transformed_Y = Y

        self.gp.fit(self.transformed_X, self.transformed_Y)

    def acq(self, X, acq):
        """Note prediction is done in normalized space.
        """

        if self.normalize_input:
            X, _, _ = zero_mean_unit_var_normalization(X, mean=self.X_mean, std=self.X_std)
        transformed_sample_predictions = self.gp.predict_all(X)

        asd = np.array([acq(pred[0, :], pred[1, :]) for pred in transformed_sample_predictions])
        # Average over all sampled hyperparameter predictions
        return np.average(asd, axis=0)

    def predict(self, X, theta=None):
        if self.normalize_input:
            X, _, _ = zero_mean_unit_var_normalization(X, mean=self.X_mean, std=self.X_std)

        mean, var = self.gp.predict(X, theta=theta)
        
        if self.normalize_output is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.y_mean, self.y_std)
            var = var * self.y_std ** 2

        return mean, var

    def plot_prediction(self, X_line, Y_line, x_new=None):
        for theta in self.gp._current_thetas:
            mean, var = self.predict(X_line, theta=theta)
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        # TODO: remember to normalize if normalization is pulled out into BOModel
        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
