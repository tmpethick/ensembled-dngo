import numpy as np
import tensorflow as tf
import GPy

GPy.plotting.change_plotting_library('matplotlib')

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean


class TFModel(object):
    def __init__(self, dim_basis=50, epochs=10000):
        self.X_mean = None
        self.X_std = None
        self.normalize_input = False

        self.y_mean = None
        self.y_std = None
        self.normalize_output = False

        self.dim_basis = dim_basis
        self.epochs = epochs

        with tf.name_scope('placeholders'):
            self.x = tf.placeholder('float', [None, 1])
            self.y_true = tf.placeholder('float', [None, 1])

        with tf.name_scope('neural_network'):
            h1 = tf.contrib.layers.fully_connected(self.x, 50, activation_fn=tf.nn.relu)
            h2 = tf.contrib.layers.fully_connected(h1, 50, activation_fn=tf.nn.relu)
            self.basis = tf.contrib.layers.fully_connected(h2, dim_basis, activation_fn=tf.nn.tanh)
            self.y_pred = tf.contrib.layers.fully_connected(self.basis, 1, activation_fn=None)
            self.loss = tf.nn.l2_loss(self.y_pred - self.y_true)

        with tf.name_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def fit(self, sess, X, y):
        # with tf.Session() as sess:
        if self.normalize_input:
            X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        
        if self.normalize_output:
            y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)

        sess.run(tf.global_variables_initializer())
        for i in range(self.epochs):
            _, train_loss = sess.run([self.train_op, self.loss],
                                      feed_dict={self.x: X,
                                                 self.y_true: y})

    def predict(self, sess, x):
        if self.normalize_input:
            x = zero_mean_unit_var_normalization(x, mean=self.X_mean, std=self.X_std)

        y_pred = sess.run(self.y_pred, {self.x: x})
        
        if self.normalize_output:
            return zero_mean_unit_var_unnormalization(y_pred, self.y_mean, self.y_std)
        else:
            return y_pred

    def predict_basis(self, sess, x):
        if self.normalize_input:
            x = zero_mean_unit_var_normalization(x, mean=self.X_mean, std=self.X_std)
        
        return sess.run(self.basis, {self.x: x})

def random_grid_samples(n_samples, bounds):
    dims = bounds.shape[0]
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dims))


class BOModel(object):
    # def __enter__(self)
    # def __exit__(self, exc_type, exc_value, traceback)

    def __init__(self, nn, n_approx_marg=0):
        # NN
        self.sess = tf.Session()
        self.nn_model = nn

        # GP
        self.kernel = GPy.kern.Linear(self.nn_model.dim_basis)

        # For marginalizing hyperparameters
        self.n_approx_marg = n_approx_marg
        self._current_thetas = None

    def init(self, X, Y):
        self.X = X
        self.Y = Y

        # NN
        self.nn_model.fit(self.sess, self.X, self.Y)
        self.D = self.nn_model.predict_basis(self.sess, self.X)

        # GP
        self.gp = GPy.models.GPRegression(self.D, self.Y, self.kernel)
        self.optimize_hyperparams()

    def add_obs(self, x_new, y_new):
        # Update data
        self.X = np.concatenate([self.X, np.array([x_new])])
        self.Y = np.concatenate([self.Y, np.array([y_new])])

        # Fit NN
        self.nn_model.fit(self.sess, self.X, self.Y)
        feature_new = self.nn_model.predict_basis(self.sess, x_new[:, None])
        self.D = np.concatenate([self.D, feature_new])

        # Fit BLR
        self.gp.set_XY(self.D, self.Y)
        self.optimize_hyperparams()

    def acq(self, X):
        beta = 16
        features = self.nn_model.predict_basis(self.sess, X)

        def _eval(theta):
            self.gp[:] = theta
            mean, std = self.gp.predict(features)
            return mean + np.sqrt(beta) * std

        if self._current_thetas is not None:
            return np.average(np.array([_eval(theta) for theta in self._current_thetas]), axis=0)
        else:
            mean, std = self.gp.predict(features)
            return mean + np.sqrt(beta) * std


    def predict(self, X):
        D = self.nn_model.predict_basis(self.sess, X)
        mean, var = self.gp.predict(D)
        return mean, var

    def plot_prediction(self, X_line, Y_line, x_new=None):
        # for f in range(min(50, model.n_units_3)):
        #     plt.plot(X_test[:, 0], basis_funcs[:, f])
        # plt.grid()
        # plt.xlabel(r"Input $x$")
        # plt.ylabel(r"Basisfunction $\theta(x)$")
        # plt.show()
        
        if self._current_thetas is not None:
            pass
            for theta in self._current_thetas:
                self.gp[:] = theta
                mean, var = self.predict(X_line)
                plt.fill_between(X_line.reshape(-1), (mean + var * 2).reshape(-1), (mean - var * 2).reshape(-1), alpha=.2)
                plt.plot(X_line, mean)
        else:
            mean, var = self.predict(X_line)
            plt.fill_between(X_line.reshape(-1), (mean + var * 2).reshape(-1), (mean - var * 2).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
        plt.plot(X_line, self.acq(X_line), color='red')
        plt.show()

    def __del__(self):
        self.sess.close()

    def optimize_hyperparams(self):
        if self.n_approx_marg > 0:
            # Most likely hyperparams given data
            hmc = GPy.inference.mcmc.HMC(self.gp)
            hmc.sample(num_samples=1000) # Burn-in
            self._current_thetas = hmc.sample(num_samples=self.n_approx_marg)

        # Optimize no matter what for plotting purposes
        self.gp.randomize()
        self.gp.optimize()


class BO(object):
    def __init__(self, obj_func, model, n_iter = 10, bounds=np.array([[0,1]])):
        self.n_iter = n_iter
        self.bounds = bounds
        self.obj_func = obj_func
        self.model = model

    def max_acq(self, n_starts=100):        
        min_y = float("inf")
        min_x = None

        def min_obj(x):
            """lift into array and negate.
            """
            x = np.array([x])
            return -self.model.acq(x)[0]

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

        self.model.plot_prediction(X_line, Y_line, x_new=x_new)

    def run(self, n_kickstart=2):
        # Data
        X = random_grid_samples(n_kickstart, self.bounds)
        Y = self.obj_func(X)
        self.model.init(X,Y)

        for i in range(0, self.n_iter):
            # new datapoint from acq
            x_new = self.max_acq()
            y_new = self.obj_func(x_new)

            # Plot
            self.plot_prediction(x_new=x_new)

            self.model.add_obs(x_new, y_new)

        self.plot_prediction()

# - works? √
# - Debug overconfidence
# - approx margin
# - to test
#     - run fit on full data √
#     - inspect nn training
#     - run bo with point estimate √
#     - run bo with sampling √
# - test on sin √ 
# - test on sinc √ 
# - test on collection
