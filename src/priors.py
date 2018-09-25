import weakref
import GPy
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from paramz.domains import _POSITIVE


class HorseshoePrior(GPy.priors.Prior):
    domain = _POSITIVE
    _instances = []

    def __new__(cls, scale=0):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().scale == scale:
                    return instance()
        newfunc = super(GPy.priors.Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, scale)     
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, scale=0.1):
        self.scale = scale

    def lnpdf(self, theta):
        if np.any(theta == 0.0):
            return np.inf
        r = np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2)
        if r == 0.0:
            return -np.inf
        return np.log(r)

    def rvs(self, n):
        lamda = np.abs(np.random.standard_cauchy(size=n))
        p0 = np.log(np.abs(np.random.randn() * lamda * self.scale))
        return p0

    def lnpdf_grad(self, theta):
        a = -(6 * self.scale ** 2)
        b = (3 * self.scale ** 2 + np.exp(2 * theta))
        b *= np.log(3 * self.scale ** 2 * np.exp(- 2 * theta) + 1)
        return a / b


class BLRPrior(object):
    def __init__(self):
        # equivalent to normal around 0
        self.ln_prior_alpha = scipy.stats.lognorm(0.1, loc=-10)

        # sigma^2 = 1 / beta
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnpdf(self, theta):
        # theta0 = ln alpha
        # theta1 = ln beta
        return self.ln_prior_alpha.logpdf(theta[0]) \
             + self.horseshoe.lnpdf(1 / np.exp(theta[1]))

    def rvs(self, num_samples):
        p0 = np.zeros([num_samples, 2])
        p0[:, 0] = self.ln_prior_alpha.rvs(num_samples)

        sigmas = self.horseshoe.rvs(num_samples) # Noise sigma^2
        p0[:, -1] = np.log(1 / np.exp(sigmas))   # Beta

        return p0
