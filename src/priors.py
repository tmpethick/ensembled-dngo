import weakref
import GPy
import numpy as np
import scipy
from scipy.special import gammaln
from scipy.stats import multivariate_normal
from paramz.domains import _REAL, _POSITIVE

class HalfT(GPy.priors.Prior):
    """
    Implementation of the half student t probability function, coupled with random variables.
    :param A: scale parameter
    :param nu: degrees of freedom
    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, A, nu):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().A == A and instance().nu == nu:
                    return instance()
        newfunc = super(GPy.priors.Prior, cls).__new__
        if newfunc is object.__new__:
              o = newfunc(cls)  
        else:
            o = newfunc(cls, A, nu)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, A, nu):
        self.A = float(A)
        self.nu = float(nu)
        self.constant = gammaln(.5*(self.nu+1.)) - gammaln(.5*self.nu) - .5*np.log(np.pi*self.A*self.nu)

    def __str__(self):
        return "hT({:.2g}, {:.2g})".format(self.A, self.nu)

    def lnpdf(self, theta):
        return (theta > 0) * (self.constant - .5*(self.nu + 1) * np.log(1. + (1./self.nu) * (theta/self.A)**2))

    def lnpdf_grad(self, theta):
        theta = theta if isinstance(theta, np.ndarray) else np.array([theta])
        grad = np.zeros_like(theta)
        above_zero = theta > 1e-6
        v = self.nu
        sigma2 = self.A
        if theta[above_zero].shape[0] > 0:
            grad[above_zero] = -0.5*(v+1)*(2*theta[above_zero])/(v*sigma2 + theta[above_zero][0]**2)
        return grad

    def rvs(self, n):
        # return np.random.randn(n) * self.sigma + self.mu
        from scipy.stats import t
        # [np.abs(x) for x in t.rvs(df=4,loc=0,scale=50, size=10000)])
        ret = t.rvs(self.nu, loc=0, scale=self.A, size=n)
        ret[ret < 0] = 0
        return ret


class GPyHorseshoePrior(GPy.priors.Prior):
    domain = _POSITIVE

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

class HorseshoePrior(object):
    domain = _REAL

    def __init__(self, scale=0.1):
        self.scale = scale

    def lnprob(self, theta):
        if np.any(theta == 0.0):
            return np.inf
        r = np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2)
        if r == 0.0:
            return -np.inf
        return np.log(r)

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
