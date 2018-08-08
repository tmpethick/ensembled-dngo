import numpy as np
import GPy
import pytest
from hpolib.benchmarks.synthetic_functions import Branin

from src.bayesian_linear_regression import BayesianLinearRegression
from src.bo import BO
from src.models import GPyBOModel, BOModel
from src.acquisition_functions import UCB
from src.neural_network import TorchRegressionModel
from src.tests import prepare_benchmark, acc_ir


@pytest.mark.skip(reason="Too slow")
def xtest_bo_gpyopt():
    import GPyOpt

    func = GPyOpt.objective_examples.experiments2d.branin()
    bounds = [{'name': 'x1', 'type': 'continuous', 'domain': func.bounds[0]},
              {'name': 'x2', 'type': 'continuous', 'domain': func.bounds[1]}]
    myBopt_mcmc = GPyOpt.methods.BayesianOptimization(func.f,
                                                      domain=bounds,
                                                      model_type='GP_MCMC',
                                                      acquisition_type='EI_MCMC',
                                                      normalize_Y=True,
                                                      n_samples=5)
    max_iter = 10
    myBopt_mcmc.run_optimization(max_iter)
    myBopt_mcmc.plot_convergence()


def test_bo_gp():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

    kernel = GPy.kern.RBF(1)
    kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)

    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=1, bounds=np.array([[0, 1]]))
    bo.run(do_plot=False)


def test_bo_gp_2d():
    import GPy
    from hpolib.benchmarks.synthetic_functions import Branin

    from src.models import GPyBOModel
    from src.acquisition_functions import UCB
    from src.bo import BO

    f, bounds, f_opt = prepare_benchmark(Branin())
    input_dim = bounds.shape[0]

    kernel = GPy.kern.RBF(input_dim, ARD=True)
    kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)

    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=1, bounds=bounds)
    bo.run(do_plot=False)


def test_bo_dngo():
    def f(x):
        return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

    nn = TorchRegressionModel(input_dim=1, dim_basis=50, epochs=100, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=0, burn_in=1000, mcmc_steps=1000)
    model = BOModel(nn, regressor=reg)

    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=1, bounds=np.array([[0,1]]))
    bo.run(do_plot=False)


def test_bo_runners():
    from src.tests import test_dngo_10_10_10_pe

    f, bounds, f_opt = prepare_benchmark(Branin())
    bo = test_dngo_10_10_10_pe(f, bounds, 1, do_plot=False)
    ir = acc_ir(bo.model.Y, f_opt)

