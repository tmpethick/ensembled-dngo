from copy import deepcopy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .bayesian_linear_regression import BayesianLinearRegression, GPyRegression
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

    def __init__(self, nn, regressor=None, num_nn=1, ensemble_aggregator=np.max, normalize_input=True, normalize_output=True):
        # NN
        self.nn_models = [deepcopy(nn) for _ in range(num_nn)]

        # reinit weights
        for m in self.nn_models:
            def weights_init(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            m.model.apply(weights_init)


        self.X_mean = None
        self.X_std = None
        self.normalize_input = normalize_input

        self.y_mean = None
        self.y_std = None
        self.normalize_output = normalize_output

        self.ensemble_aggregator = ensemble_aggregator
        self.num_nn = num_nn

        # GP
        if regressor is None:
            self.gps = [BayesianLinearRegression(num_mcmc=0) for _ in range(num_nn)]
        else:
            self.gps = [deepcopy(regressor) for _ in range(num_nn)]

    def fit(self, X, Y, train_nn=True):
        if self.normalize_input:
            self.transformed_X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.transformed_X = X

        if self.normalize_output:
            self.transformed_Y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(Y)
        else:
            self.transformed_Y = Y

        for i, nn in enumerate(self.nn_models):
            if train_nn:
                # NN
                nn.fit(self.transformed_X, self.transformed_Y)
                transformed_D = nn.predict_basis(self.transformed_X)

                self.gps[i].fit(transformed_D, self.transformed_Y)

    def acq(self, X, acq):
        """Note: prediction is done in normalized space.
        """

        Ds = self.predict_basis(X)

        idxs = np.arange(self.num_nn)
        def predict_all(i):
            return self.gps[i].predict_all(Ds[i])
        predict_all = np.vectorize(predict_all, signature='()->(m,t,n)')

        transformed_sample_predictions = predict_all(idxs)
        # shape: (ensemble, gphyperparams, summarystats, samples)

        # Calc acq
        acq_values = np.apply_along_axis(lambda x: acq(x[0], x[1]), 2, transformed_sample_predictions)
        # shape: (ensemble, gphyperparams, samples)

        # Average over all sampled hyperparameter predictions
        hyper_average = np.average(acq_values, 1)
        # shape: (ensemble, samples)

        # Second aggregate ensemble
        ensemble_agg = self.ensemble_aggregator(hyper_average, axis=0)
        return ensemble_agg

    def predict_basis(self, X):
        """ 
        return -- shape: (ensemble, samples, basisfunctions)
        """
        if self.normalize_input:
            X, _, _ = zero_mean_unit_var_normalization(X, mean=self.X_mean, std=self.X_std)

        idxs = np.arange(self.num_nn)
        def _predict_basis(i):
            return self.nn_models[i].predict_basis(X)
        return np.vectorize(_predict_basis, signature='()->(m,n)')(idxs)
        
    def predict_from_basis(self, transformed_D, theta=None):
        """
        transformed_D -- shape: (ensemble, samples, basisfunctions)
        return        -- shape: (ensemble, gphyperparams, summarystats, samples)
        """

        idxs = np.arange(self.num_nn)
        def _predict_from_basis(i):
            return self.gps[i].predict_all(transformed_D[i])
        predictions = np.vectorize(_predict_from_basis, signature='()->(h,s,n)')(idxs)
        # (ensemble, hyperparams, summarystats, samples)

        if self.normalize_output is not None:
            def normalize(summ):
                mean = zero_mean_unit_var_unnormalization(summ[0], self.y_mean, self.y_std)
                var = summ[1] * self.y_std ** 2
                return np.array([mean[0], var[0]])

            predictions = np.apply_along_axis(normalize, 2, predictions)

        return predictions

    def predict(self, X):
        D = self.predict_basis(X)
        return self.predict_from_basis(D)

    def plot_prediction(self, X_line, Y_line, x_new=None):
        D_line = self.predict_basis(X_line)


        predictions = self.predict_from_basis(D_line) # shape: (ensemble, gphyperparams, summ, samples)
        for summ in predictions.reshape(-1, predictions.shape[2], predictions.shape[3]): # (models, summ, samples)
            mean = summ[0, :]
            var = summ[1, :]
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

        return np.stack([mean, var])

    def plot_prediction(self, X_line, Y_line, x_new=None):
        for theta in self.gp._current_thetas:
            summ = self.predict(X_line, theta=theta)
            mean = summ[0]
            var = summ[1]
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        # TODO: remember to normalize if normalization is pulled out into BOModel
        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
