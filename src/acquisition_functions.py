import numpy as np
import scipy


class EI(object):
    def __init__(self, model, par=0.01):
        self.model = model
        self.par = par

    def calc(self, mean, var):
        _, eta = self.model.get_incumbent()
        s = np.sqrt(var)

        if (s == 0).any():
            return np.array([[0]])
        else:
            z = (mean - eta - self.par) / s
            return (mean - eta - self.par) * scipy.stats.norm.cdf(z) \
                   + s * scipy.stats.norm.pdf(z)


class UCB(object):
    def __init__(self, model):
        self.model = model

    def calc(self, mean, var):
        beta = 16
        return mean + np.sqrt(beta) * np.sqrt(var)
