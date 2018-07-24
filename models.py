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

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean

# class GPyLinearRegression(object):
#     def __init__(self, alpha=1, beta=1000, prior=None):
#     def marginal_log_likelihood(self, theta):
#     def negative_mll(self, theta):
#     def fit(self, X, y, do_optimize=True):
#     def predict(self, X):

class HorseshoePrior(Prior):
    domain = _REAL

    def __init__(self, scale=0.1):
        self.scale = scale

    def lnprob(self, theta):
        if np.any(theta == 0.0):
            return np.inf
        return np.log(np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2))

    def lnpdf(self, x):
        return self.lnprob(x)

class BLRPrior(object):
    def __init__(self):
        self.ln_prior_alpha = scipy.stats.lognorm(0.1, loc=-10)
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        return self.ln_prior_alpha.logpdf(theta[0]) \
             + self.horseshoe.lnprob(1 / theta[-1])


class BayesianLinearRegression(object):
    def __init__(self, alpha=1, beta=1000, prior=BLRPrior()):
        self.alpha = alpha
        self.beta = beta
        
        self.prior = prior

    def marginal_log_likelihood(self, theta):
        if np.any((-5 > theta) + (theta > 10)):
            return -1e25

        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        D = self.X.shape[1]
        N = self.X.shape[0]

        K = beta * np.dot(self.X.T, self.X)
        K += np.eye(self.X.shape[1]) * alpha
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.X.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.X, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K))
        
        if self.prior is not None:
            l = mll + self.prior.lnprob(theta)
            return l
        else:
            return mll

    def negative_mll(self, theta):
        return -self.marginal_log_likelihood(theta)

    # def marginal_log_likelihood(self, theta):
    #     # Theta is on a log scale
    #     alpha = np.exp(theta[0])
    #     beta = np.exp(theta[1])

    #     D = self.X.shape[1]
    #     N = self.X.shape[0]

    #     A = beta * np.dot(self.X.T, self.X)
    #     A += np.eye(self.X.shape[1]) * alpha
    #     A_inv = np.linalg.inv(A)
    #     m = beta * np.dot(A_inv, self.X.T)
    #     m = np.dot(m, self.y)

    #     mll = D / 2 * np.log(alpha)
    #     mll += N / 2 * np.log(beta)
    #     mll -= N / 2 * np.log(2 * np.pi)
    #     mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.X, m), 2)
    #     mll -= alpha / 2. * np.dot(m.T, m)
    #     mll -= 0.5 * np.log(np.linalg.det(A))

    #     if self.prior is not None:
    #         mll += self.prior.lnprob(theta)

    #     return mll

    # def negative_mll(self, theta):
    #     return -self.marginal_log_likelihood(theta)

    def fit(self, X, y, do_optimize=True):
        self.X = X
        self.y = y

        if do_optimize:
            res = optimize.fmin(self.negative_mll, np.random.rand(2))
            self.alpha = np.exp(res[0])
            self.beta = np.exp(res[1])

        S_inv = self.beta * np.dot(self.X.T, self.X)
        S_inv += np.eye(self.X.shape[1]) * self.alpha

        S = np.linalg.inv(S_inv)
        m = self.beta * np.dot(np.dot(S, self.X.T), self.y)

        self.m = m 
        self.S = S

    def predict(self, X):        
        m = np.dot(self.m.T, X.T)
        v = np.diag(np.dot(np.dot(X, self.S), X.T)) + 1. / self.beta
        m = m.T
        v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        v = v[:, None]
        return m, v

class TFModel(object):
    def __init__(self, dim_basis=50, epochs=10000, batch_size=10):
        self.X_mean = None
        self.X_std = None
        self.normalize_input = True

        self.y_mean = None
        self.y_std = None
        self.normalize_output = True

        self.dim_basis = dim_basis
        self.epochs = epochs
        self.batch_size = batch_size

        with tf.name_scope('placeholders'):
            self.x = tf.placeholder('float', [None, 1])
            self.y_true = tf.placeholder('float', [None, 1])

        with tf.name_scope('neural_network'):
            h1 = tf.contrib.layers.fully_connected(self.x, 50, 
                            activation_fn=tf.nn.tanh)
                            # # biases_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            # # weights_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            h2 = tf.contrib.layers.fully_connected(h1, 50, 
                            activation_fn=tf.nn.tanh) 
                            # # biases_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            # # weights_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.basis = tf.contrib.layers.fully_connected(h2, dim_basis, 
                            activation_fn=tf.nn.tanh) 
                            # # biases_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            # # weights_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.y_pred = tf.contrib.layers.fully_connected(self.basis, 1, activation_fn=None)
            self.loss = tf.nn.l2_loss(self.y_pred - self.y_true)
            # self.loss = tf.divide(tf.reduce_mean(tf.square(self.y_pred - self.y_true)), 0.001)

        with tf.name_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def fit(self, sess, X, y):
        # with tf.Session() as sess:
        if self.normalize_input:
            X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        
        if self.normalize_output:
            y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)

        self.X = X
        self.y = y

        sess.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            for input_batch, output_batch in self.iterate_minibatches(X, y, shuffle=True):
                _, train_loss = sess.run([self.train_op, self.loss],
                                        feed_dict={self.x: X,
                                                    self.y_true: y})

    def iterate_minibatches(self, inputs, targets, shuffle=True):
        assert inputs.shape[0] == targets.shape[0],\
               "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield inputs[excerpt], targets[excerpt]

    def predict(self, sess, x):
        if self.normalize_input:
            x, _, _ = zero_mean_unit_var_normalization(x, mean=self.X_mean, std=self.X_std)

        y_pred = sess.run(self.y_pred, {self.x: x})
        
        if self.normalize_output:
            return zero_mean_unit_var_unnormalization(y_pred, self.y_mean, self.y_std)
        else:
            return y_pred

    def predict_basis(self, sess, x):
        if self.normalize_input:
            x, _, _ = zero_mean_unit_var_normalization(x, mean=self.X_mean, std=self.X_std)
        
        return sess.run(self.basis, {self.x: x})

def random_grid_samples(n_samples, bounds):
    dims = bounds.shape[0]
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dims))


class BOModel(object):
    # def __enter__(self)
    # def __exit__(self, exc_type, exc_value, traceback)

    def __init__(self, nn, n_approx_marg=0, use_gpy=True):
        # NN
        self.sess = tf.Session()
        self.nn_model = nn
        self.use_gpy = use_gpy

        # For marginalizing hyperparameters
        self.n_approx_marg = n_approx_marg
        self._current_thetas = None

    def init(self, X, Y, train_nn=True):
        self.X = X
        self.Y = Y

        # NN
        if train_nn:
            self.nn_model.fit(self.sess, self.X, self.Y)
        self.D = self.nn_model.predict_basis(self.sess, self.X)

        # GP
        if self.use_gpy:
            # GP
            self.kernel = GPy.kern.Linear(self.nn_model.dim_basis)        
            self.gp = GPy.models.GPRegression(self.D, self.nn_model.y, self.kernel)
            
            # Set hyperpriors
            hyperprior = GPy.priors.Gamma.from_EV(0.5, 1)
            self.kernel.variances.set_prior(hyperprior) # log_prior()
            self.gp.Gaussian_noise.variance.set_prior(hyperprior)
        else:
            self.gp = BayesianLinearRegression(self.D, self.nn_model.y)
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
        # self.gp.fit(self.D, self.Y, use_mcmc=self.n_approx_marg > 0)
        self.gp.set_XY(self.D, self.nn_model.y)
        self.optimize_hyperparams()

    def acq(self, X):
        # TODO: keep output normalize
        beta = 16
        features = self.nn_model.predict_basis(self.sess, X)

        def _eval(theta):
            self.gp[:] = theta
            mean, std = self.predict_from_basis(features)
            return mean + np.sqrt(beta) * std

        return np.average(np.array([_eval(theta) for theta in self._current_thetas]), axis=0)

    def predict_from_basis(self, D):
        mean, var = self.gp.predict(D)

        if self.nn_model.y_std is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.nn_model.y_mean, self.nn_model.y_std)
            var = var * self.nn_model.y_std ** 2

        return mean, var

    def predict(self, X):
        D = self.nn_model.predict_basis(self.sess, X)
        return self.predict_from_basis(D)

    def plot_prediction(self, X_line, Y_line, x_new=None):        
        for theta in self._current_thetas:
            self.gp[:] = theta
            mean, var = self.predict(X_line)
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)

        if x_new is not None:
            plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')

        # TODO: remember to normalize if normalization is pulled out into BOModel
        plt.scatter(self.X, self.Y)
        plt.plot(X_line, Y_line, dashes=[2, 2], color='black')
        # plt.plot(X_line, self.acq(X_line), color='red')
        plt.show()

    def __del__(self):
        self.sess.close()

    def optimize_hyperparams(self):
        if self.n_approx_marg > 0:
            # Most likely hyperparams given data
            hmc = GPy.inference.mcmc.HMC(self.gp)
            hmc.sample(num_samples=2000) # Burn-in
            self._current_thetas = hmc.sample(num_samples=2, hmc_iters=50)
        else:
            if self.use_gpy:
                self.gp.randomize()
                self.gp.optimize()
                self._current_thetas = [self.gp.param_array]
            else:
                self._current_thetas = [self.gp.param_array]


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
