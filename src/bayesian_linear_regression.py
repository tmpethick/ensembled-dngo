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
import emcee

from .priors import BLRPrior

class BayesianLinearRegression(object):
    def __init__(self, alpha=1, beta=1000, prior=None, num_mcmc=4, burn_in=1000, mcmc_steps=1000, do_optimize=True):
        self.alpha = alpha
        self.beta = beta
        
        self.do_optimize = do_optimize
        self.num_mcmc = num_mcmc
        self.burn_in = burn_in
        self.mcmc_steps = mcmc_steps
        
        self.prior = prior if prior is not None else BLRPrior()

    def marginal_log_likelihood(self, theta):
        """Note that theta is transformed using log scale.
        Take exp to get the real parameters.
        """

        # Contrain in between 0.00673 and 22026
        if np.any((-5 > theta) + (theta > 10)):
            return -np.float("inf")

        t = self.y
        Theta = self.X

        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        M = Theta.shape[1]
        N = Theta.shape[0]

        A = beta * np.dot(Theta.T, Theta) + np.eye(M) * alpha # (3.81)
        A_inv = np.linalg.inv(A)
        m = np.dot(beta * np.dot(A_inv, Theta.T), t)          # (3.53)

        # TODO: understand why not always positive
        detA = np.linalg.det(A)
        if detA <= 0.0:
            return -np.float("inf")

        # (3.86)
        mll  = M / 2 * np.log(alpha)          #   M/2 ln alpha
        mll += N / 2 * np.log(beta)           # + N/2 ln beta
        mll -= N / 2 * np.log(2 * np.pi)      # - N/2 ln (2 pi)
        mll -= beta / 2. * np.linalg.norm(t - np.dot(Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)    # - E(mN) (3.82)
        mll -= 0.5 * np.log(np.linalg.det(A)) # - 1/2 ln |A|
        
        l = mll + self.prior.lnprob(theta)
        return l

    def negative_mll(self, theta):
        return -self.marginal_log_likelihood(theta)

    def fit(self, X, y):
        self.X = X
        self.y = y

        def copy_model(alpha, beta):
            m = BayesianLinearRegression(alpha, beta, do_optimize=False)
            m.fit(self.X, self.y)
            return m

        if self.do_optimize:
            if self.num_mcmc > 0:
                ndim, nwalkers = 2, self.num_mcmc
                p0 = [np.random.rand(ndim) for i in range(nwalkers)] # self.prior.rvs(nwalkers)

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.marginal_log_likelihood)
                sampler.run_mcmc(p0, self.burn_in + self.mcmc_steps)
                hyper_pairs = np.exp(sampler.chain[:, -1, :])
                self._hypers = hyper_pairs
                # TODO: rename to models
                self._current_thetas = [copy_model(alpha, beta) for alpha, beta in hyper_pairs]
            else:
                res = optimize.fmin(self.negative_mll, np.random.rand(2), disp=False)
                self.alpha = np.exp(res[0])
                self.beta = np.exp(res[1])

        S_inv = self.beta * np.dot(self.X.T, self.X) + np.eye(self.X.shape[1]) * self.alpha

        # TODO: use cholesky to make numerically stable
        S = np.linalg.inv(S_inv)
        m = self.beta * np.dot(np.dot(S, self.X.T), self.y)

        self.m = m 
        self.S = S

    def predict_all(self, X):
        return np.array([self.predict(X)])

    def predict(self, X, theta=None):
        if theta is not None:
            model = theta
            return model.predict(X)

        m = np.dot(self.m.T, X.T)
        v = np.diag(np.dot(np.dot(X, self.S), X.T)) + 1. / self.beta
        m = m.T
        v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        v = v[:, None]
        return m, v

# class GPyLinearRegression(GPyRegression):
#     def __init__(**kwargs):
#         super().__init__(kernel=GPy.kern.Linear(dim_basis), **kwargs)

# class GPyRegression(object):
#     def __init__(self, kernel, num_mcmc=0, do_optimize=True):
#         self._current_thetas = None

#         # If 0 max point estimate is used via max likelihood.
#         self.num_mcmc = num_mcmc
#         self.do_optimize = do_optimize
#         self.kernel = kernel

#     def fit(self, X, y):
#         dim_basis = X.shape[1]
#         self.gp = GPy.models.GPRegression(X, y, self.kernel)

#         # Set hyperpriors
#         hyperprior = GPy.priors.Gamma.from_EV(0.5, 1)
#         self.kernel.variances.set_prior(hyperprior) # log_prior()
#         self.gp.Gaussian_noise.variance.set_prior(hyperprior)

#         # Optimize
#         if self.do_optimize:
#             if self.num_mcmc > 0:
#                 # Most likely hyperparams given data
#                 hmc = GPy.inference.mcmc.HMC(self.gp)
#                 hmc.sample(num_samples=2000) # Burn-in
#                 self._current_thetas = hmc.sample(num_samples=2, hmc_iters=50)
#             else:
#                 self.gp.randomize()
#                 self.gp.optimize()
#                 self._current_thetas = [self.gp.param_array]        

#     def predict_all(self, X):
#         predictions = np.zeros((len(self._current_thetas), 2))
#         for i, theta in enumerate(self._current_thetas):
#             self.gp[:] = self._current_thetas[0]
#             predictions[i, :] = self.gp.predict(X)
#         return predictions

#     def predict(self, X, theta=None):
#         # TODO: use aggregate of samples
#         if theta is not None:
#             self.gp[:] = theta
#         return self.gp.predict(X)
