import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


class TFModel(object):
    def __init__(self, input_dim=1, dim_basis=50, dim_h1=50, dim_h2=50, epochs=10000, batch_size=10):
        self.dim_basis = dim_basis
        self.epochs = epochs
        self.batch_size = batch_size

        with tf.name_scope('placeholders'):
            self.x = tf.placeholder('float', [None, input_dim])
            self.y_true = tf.placeholder('float', [None, 1])

        with tf.name_scope('neural_network'):
            h1 = tf.contrib.layers.fully_connected(self.x, dim_h1,
                                                   activation_fn=tf.nn.tanh)
            # # biases_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            # # weights_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            h2 = tf.contrib.layers.fully_connected(h1, dim_h2,
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
        sess.run(tf.global_variables_initializer())

        batch_size = min(X.shape[0], self.batch_size)

        for i in range(self.epochs):
            for input_batch, output_batch in self.iterate_batches(X, y, shuffle=True, batch_size=batch_size):
                _, train_loss = sess.run([self.train_op, self.loss],
                                         feed_dict={self.x: input_batch,
                                                    self.y_true: output_batch})

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
        return sess.run(self.y_pred, {self.x: x})

    def predict_basis(self, sess, x):
        return sess.run(self.basis, {self.x: x})

    def plot_basis_functions(self, sess, bounds):
        if bounds.shape is not (1, 2):
            raise Exception("Only supports 1D")

        x = np.linspace(bounds[:, 0], bounds[:, 1])[:, None]
        D = self.predict_basis(sess, x)
        for i in range(self.dim_basis):
            plt.plot(x, D[:, i])
