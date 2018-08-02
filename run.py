
# from src.tests import *
# f, bounds, f_opt = prepare_benchmark(Branin())
# # bo = test_gp(f, bounds, 20, do_plot=False)
# # bo = test_dngo_50_50_50_marg(f, bounds, 20, do_plot=False)
# bo = test_dngo_10_10_10_pe(f, bounds, 10, do_plot=True)
# ir = acc_ir(bo.model.Y, f_opt)



# import GPy
# import numpy as np
# import matplotlib.pyplot as plt
#
# from src.bayesian_linear_regression import BayesianLinearRegression
# from src.models import GPyBOModel, BOModel
# from src.neural_network import NNRegressionModel
#
# def f(x):
#     return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000
#
#
# rng = np.random.RandomState(42)
# x_train = rng.uniform(0, 1, 10)[:, None]
# y_train = f(x_train)
#
# nn = NNRegressionModel(dim_basis=50, epochs=1000, batch_size=10)
# reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
# model = BOModel(nn, regressor=reg,  num_nn=5)
# model.init(x_train, y_train)
#
# x = (np.linspace(0, 1, 100))[:, None]
# y = f(x)
# model.plot_prediction(x, y)
# plt.show(block=True)


# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from plotly.offline import iplot

from src.tests import *
from hpolib.benchmarks.synthetic_functions import Branin
f, bounds, f_opt = prepare_benchmark(Branin())

boo = test_dngo_10_10_10_pe(f, bounds, 3, do_plot=True)

import tensorflow as tf
boo.model.sess = tf.Session()
model = boo.model
model.fit(model.X, model.Y)
fig = boo.plot_2D_surface(use_plotly=True)
iplot(fig)