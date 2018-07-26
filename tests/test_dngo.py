import numpy as np
import matplotlib.pyplot as plt

from src.bo import *
from src.dngo import *

def test_dngo_MAP():  
  def f(x):
      return np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000

  rng = np.random.RandomState(42)
  x_train = rng.uniform(0, 1, 10)[:, None]
  y_train = f(x_train)

  nn = TFModel(dim_basis=50, epochs=1000, batch_size=10)
  reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
  model = BOModel(nn, regressor=reg)
  model.init(x_train, y_train)

  x = (np.linspace(0, 1, 100))[:,None]
  y = f(x)
  model.plot_prediction(x,y)
  plt.show(block=True)

def test_dngo_approximate_marginalisation_of_hyperparameters():  
  def f(x):
      return np.sinc(x * 10 - 5).sum(axis=1)[:, None] # * 100000

  rng = np.random.RandomState(42)
  x_train = rng.uniform(0, 1, 10)[:, None]
  y_train = f(x_train)

  nn = TFModel(dim_basis=50, epochs=1000, batch_size=10)
  reg = BayesianLinearRegression(num_mcmc=20, burn_in=10, mcmc_steps=10)
  model = BOModel(nn, regressor=reg)
  model.init(x_train, y_train)

  x = (np.linspace(0, 1, 100))[:,None]
  y = f(x)
  model.plot_prediction(x,y)
  plt.show(block=True)
