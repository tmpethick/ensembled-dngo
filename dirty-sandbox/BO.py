#%%
import numpy as np
import GPy
GPy.plotting.change_plotting_library('matplotlib')

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

class GBUCB(object):
    def __init__(self, beta = lambda t: 2):
        self.get_beta = beta

    def update_model(self, m, theta):
        m[:] = theta
        # for i, param_name in enumerate(m.parameter_names()):
        #    m[param_name] = theta[i]

    def eval(self, x, t, m, thetas):
        """`x` is (n,d)
        """
        def _eval(theta):
            self.update_model(m, theta)
            beta_t = self.get_beta(t)
            mean, std = m.predict(x)
            # shape: (n, yd)
            # average over yd
            return np.average(mean + np.sqrt(beta_t) * std, axis=1)

        return np.average(np.array([_eval(theta) for theta in thetas]), axis=0)

class BO(object):
    """
    - take hyperprior (use GP order)
    - draw thetas from p(theta|D): mcmc log_likelihood + hyperprior
    - maximize GP-UCB
    - set parameters on m (copy ideally)
    """

    def __init__(self, obj_func, kernel, acquisition, bounds, M=50, hyperparam_point_estimate=False):
        self.obj_func = obj_func
        self.M = M
        self.kernel = kernel
        self.bounds = bounds
        self.acquisition = acquisition
        self.hyperparam_point_estimate = hyperparam_point_estimate

        self._current_thetas = None
        self.m = None
        self.t = None

    def sample_thetas(self):
        if self.hyperparam_point_estimate:
            # Reset parameters before optimizing to prevent being stuck in local minimum.
            self.m[:] = np.random.normal(self.m.size)
            
            self.m.optimize(messages=True)
            return np.array([self.m.param_array])
        else:
            hmc = GPy.inference.mcmc.HMC(self.m)
            hmc.sample(num_samples=1000) # Burn-in
            return hmc.sample(num_samples=self.M)
            
    
    def grid_samples(self, n_samples):
        dims = self.bounds.shape[0]
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_samples, dims))

    def random_grid_samples(self, n_samples):
        dims = self.bounds.shape[0]
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_samples, dims))

    def max_average_acquisition(self, t, n_starts=10):
        min_y = float("inf")
        min_x = None

        thetas = self.sample_thetas()
        self._current_thetas = thetas
        
        def min_obj(x):
            """lift into array and negate.
            """
            x = np.array([x])
            return -self.acquisition.eval(x, t, self.m, thetas)[0]

        # TODO: turn into numpy operations to parallelize
        for x0 in self.random_grid_samples(n_starts):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 
        
        return min_x


    def find_next(self, t):
        return self.max_average_acquisition(t)

    def update_posterior(self, xnew):
        ynew = self.obj_func(xnew)
        X = np.concatenate([self.m.X, np.array([xnew])])
        Y = np.concatenate([self.m.Y, np.array([ynew])])
        self.m.set_XY(X, Y)

    def run(self, n_iter=10, plot=True, true_func=None):
        x0 = self.random_grid_samples(2)
        y0 = self.obj_func(x0)

        self.m = GPy.models.GPRegression(x0, y0, kernel=self.kernel, noise_var=0.01)

        # set hyperprior
        if not self.hyperparam_point_estimate:
            hyperprior = GPy.priors.Gamma.from_EV(0.5, 1)
            self.m.kern.lengthscale.set_prior(hyperprior)
            self.m.kern.variance.set_prior(hyperprior)
            self.m.Gaussian_noise.variance.set_prior(hyperprior)

        if plot:
            # required to plot acquisition func
            boundedX = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)

        t = 0
        for i in range(0, n_iter):
            xnew = self.find_next(t)
            t += 1

            if plot:
                plt.figure()
                acY = self.acquisition.eval(boundedX, t, self.m, self._current_thetas)
                plt.scatter(self.m.X, self.m.Y)
                if true_func is not None:
                    plt.plot(boundedX, true_func(boundedX), 'b-', lw=1, label='True function')
                plt.plot(boundedX, acY, 'r-', lw=1, label='Acquisition function')
                plt.axvline(x=xnew, ls='--', c='k', lw=1, label='Next sampling location')
                plt.show()
                
            self.update_posterior(xnew)
        return xnew

