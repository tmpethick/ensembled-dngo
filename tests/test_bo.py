import GPy

from src.bo import BO
from src.dngo import GPyBOModel
from src.acquisition_functions import EI, UCB

from src.bo import *
from src.dngo import *


def test_bo_gp():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

    kernel = GPy.kern.RBF(1)
    # kernel.variance.set_prior(GPy.priors.LogGaussian(0.005, 1)) # log_prior()
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)

    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=1, bounds=np.array([[0,1]]))
    bo.run(do_plot=False)

def test_bo_dngo():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

    nn = TFModel(input_dim=1, dim_basis=50, epochs=100, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
    model = BOModel(nn, regressor=reg)

    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=1, bounds=np.array([[0,1]]))
    bo.run(do_plot=False)
