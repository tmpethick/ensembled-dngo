import GPy
import numpy as np
import matplotlib.pyplot as plt

from src.bayesian_linear_regression import BayesianLinearRegression
from src.models import GPyBOModel, BOModel
from src.neural_network import NNRegressionModel


def test_gp():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000

    rng = np.random.RandomState(42)
    x_train = rng.uniform(0, 1, 10)[:, None]
    y_train = f(x_train)

    kernel = GPy.kern.RBF(1)
    # kernel.variance.set_prior(GPy.priors.LogGaussian(0.005, 1)) # log_prior()
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)
    model.init(x_train, y_train)

    x = (np.linspace(0, 1, 100))[:,None]
    y = f(x)
    model.plot_prediction(x,y)
    plt.show(block=True)


def test_dngo_MAP():  
  def f(x):
      return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000

  rng = np.random.RandomState(42)
  x_train = rng.uniform(0, 1, 10)[:, None]
  y_train = f(x_train)

  nn = NNRegressionModel(dim_basis=50, epochs=1000, batch_size=10)
  reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
  model = BOModel(nn, regressor=reg)
  model.init(x_train, y_train)

  x = (np.linspace(0, 1, 100))[:,None]
  y = f(x)
  model.plot_prediction(x,y)
  plt.show(block=True)


def test_dngo_approximate_marginalisation_of_hyperparameters():  
  def f(x):
      return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000

  rng = np.random.RandomState(42)
  x_train = rng.uniform(0, 1, 10)[:, None]
  y_train = f(x_train)

  nn = NNRegressionModel(dim_basis=50, epochs=1000, batch_size=10)
  reg = BayesianLinearRegression(num_mcmc=20, burn_in=10, mcmc_steps=10)
  model = BOModel(nn, regressor=reg)
  model.init(x_train, y_train)

  x = (np.linspace(0, 1, 100))[:,None]
  y = f(x)
  model.plot_prediction(x,y)
  plt.show(block=True)

def test_dngo_ensemble():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000

    rng = np.random.RandomState(42)
    x_train = rng.uniform(0, 1, 10)[:, None]
    y_train = f(x_train)

    nn = NNRegressionModel(dim_basis=50, epochs=1000, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
    model = BOModel(nn, regressor=reg, num_nn=5)
    model.init(x_train, y_train)

    x = (np.linspace(0, 1, 100))[:, None]
    y = f(x)
    model.plot_prediction(x, y)
    plt.show(block=True)
