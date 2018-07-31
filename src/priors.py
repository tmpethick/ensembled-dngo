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

class HorseshoePrior(object):
    domain = _REAL

    def __init__(self, scale=0.1):
        self.scale = scale

    def lnprob(self, theta):
        if np.any(theta == 0.0):
            return np.inf
        return np.log(np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2))

    def rvs(self, num_samples):
        lamda = np.abs(np.random.standard_cauchy(size=num_samples))
        p0 = np.log(np.abs(np.random.randn() * lamda * self.scale))
        return p0

class BLRPrior(object):
    def __init__(self):
        # equivalent to normal around 0
        self.ln_prior_alpha = scipy.stats.lognorm(0.1, loc=-10)

        # sigma^2 = 1 / beta
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        # theta0 = ln alpha
        # if X is log normal then Y = lg X is normal.. 
        # lg (X) is normal

        # theta1 = ln beta
        return self.ln_prior_alpha.logpdf(theta[0]) \
             + self.horseshoe.lnprob(1 / np.exp(theta[1]))

    def rvs(self, num_samples):
        p0 = np.zeros([num_samples, 2])
        p0[:, 0] = self.ln_prior_alpha.rvs(num_samples)

        sigmas = self.horseshoe.rvs(num_samples) # Noise sigma^2
        p0[:, -1] = np.log(1 / np.exp(sigmas))   # Beta

        return p0
