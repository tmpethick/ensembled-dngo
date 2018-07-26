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

class TFModel(object):
    def __init__(self, input_dim=1, dim_basis=50, epochs=10000, batch_size=10):
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
            self.x = tf.placeholder('float', [None, input_dim])
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

        batch_size = min(self.X.shape[0], self.batch_size)

        for i in range(self.epochs):
            for input_batch, output_batch in self.iterate_batches(X, y, shuffle=True, batch_size=batch_size):
                _, train_loss = sess.run([self.train_op, self.loss],
                                        feed_dict={self.x: X,
                                                    self.y_true: y})

    def iterate_batches(self, inputs, targets, shuffle=True, batch_size=1):
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
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

    def plot_basis_functions(self, sess, x):
        D = self.predict_basis(sess, x)
        for i in range(self.dim_basis):
            plt.plot(x, D[:, i])
