from functools import wraps
from operator import itemgetter

import GPy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathos.multiprocessing as mp
# import torch.multiprocessing as mp
from hpolib.benchmarks.synthetic_functions import Branin

from src.utils import random_hypercube_samples, vectorize


def immidiate_regret(y, y_opt):
    return np.abs(y - y_opt)


def acc_ir(history, y_opt):
    max_history = -np.maximum.accumulate(history)
    regret_history = immidiate_regret(max_history, y_opt)
    return regret_history


def plot_ir(acc_ir_arrays):
    for acc_ir in acc_ir_arrays:
        plt.plot(acc_ir)
    plt.yscale('log')


def prepare_benchmark(func):
    """"
    Arguments:
        Func -- Function from hpolib.benchmarks.synthetic_functions
    
    Returns:
        (function, np array of bounds, minimum f value)
    """

    info = func.get_meta_information()
    bounds = np.array(info['bounds'])

    @wraps(func)
    def wrapper(x):
        return -func(x)
    
    return vectorize(wrapper), bounds, info['f_opt']


def test_random_sample(f, bounds, n_iter=100):
    # Random sample
    R_samples = random_hypercube_samples(n_iter, bounds)
    R_values = f(R_samples)
    # R_values[np.argmax(R_values)]
    return R_samples, R_values


def test_gp(f, bounds, n_iter, do_plot=False):
    import GPy
    from .models import GPyBOModel
    from .acquisition_functions import UCB
    from .bo import BO

    input_dim = bounds.shape[0]

    kernel = GPy.kern.RBF(input_dim)
    kernel.variance.set_prior(GPy.priors.LogGaussian(0.005, 0.5)) # log_prior()
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)

    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    return bo


def dngo_test_factory(dim_basis=10, dim_h1=10, dim_h2=10, num_mcmc=0, num_nn=1, batch_size=1000, epochs=1000, lr=0.01, weight_decay=0):
    def test_dngo(f, bounds, n_iter, do_plot=False):
        from .bo import BO
        from .acquisition_functions import EI, UCB
        from .bayesian_linear_regression import BayesianLinearRegression, GPyRegression
        from .models import BOModel
        from .neural_network import TorchRegressionModel

        input_dim = bounds.shape[0]
        nn = TorchRegressionModel(input_dim=input_dim, dim_basis=dim_basis, dim_h1=dim_h1, dim_h2=dim_h2, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay)
        # reg = BayesianLinearRegression(num_mcmc=num_mcmc)
        kernel = GPy.kern.Linear(dim_basis)
        kernel.variances.set_prior(GPy.priors.LogGaussian(0, 1))
        reg = GPyRegression(kernel=kernel, num_mcmc=num_mcmc, fix_noise=True)

        model = BOModel(nn, regressor=reg, num_nn=num_nn)
        # acq = EI(model, par=0.01)
        acq = UCB(model)
        bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
        bo.run(do_plot=do_plot)
        return bo
    return test_dngo

def test_dngo_10_10_10_marg(f, bounds, n_iter, do_plot=False):
    from .bo import BO
    from .acquisition_functions import EI, UCB
    from .bayesian_linear_regression import BayesianLinearRegression
    from .models import BOModel
    from .neural_network import TorchRegressionModel

    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=6)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo

def test_dngo_10_10_10_pe_ensemble(f, bounds, n_iter, do_plot=False):
    from .bo import BO
    from .acquisition_functions import EI, UCB
    from .bayesian_linear_regression import BayesianLinearRegression
    from .models import BOModel
    from .neural_network import TorchRegressionModel

    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = BOModel(nn, regressor=reg, num_nn=5)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo


def test_dngo_10_10_10_pe(f, bounds, n_iter, do_plot=False):
    from .bo import BO
    from .acquisition_functions import EI, UCB
    from .bayesian_linear_regression import BayesianLinearRegression
    from .models import BOModel
    from .neural_network import TorchRegressionModel

    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo

def test_dngo_50_50_50_pe(f, bounds, n_iter, do_plot=False):

    from .bo import BO
    from .acquisition_functions import EI, UCB
    from .bayesian_linear_regression import BayesianLinearRegression
    from .models import BOModel
    from .neural_network import TorchRegressionModel

    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo


def test_dngo_50_50_50_marg(f, bounds, n_iter, do_plot=False):
    from .bo import BO
    from .acquisition_functions import EI, UCB
    from .bayesian_linear_regression import BayesianLinearRegression
    from .models import BOModel
    from .neural_network import TorchRegressionModel

    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=20, burn_in=1000, mcmc_steps=1000)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)

    return bo

def test_method(bo_method, n_iter=200, Benchmark=None, do_plot=False):
    benchmark = Benchmark() if Benchmark is not None else Branin()
    f, bounds, f_opt = prepare_benchmark(benchmark)
    bo = bo_method(f, bounds, n_iter, do_plot=do_plot)
    ir = acc_ir(bo.model.Y, f_opt)

    sns.set_style("darkgrid")
    plot_ir([ir])

    return bo

def test_multiple(bo_methods, n_iter=200, Benchmark=None):
    """
    Arguments:
        bo_methods -- dictionary of functions returning BOBaseModel
    """

    benchmark = Benchmark() if Benchmark is not None else Branin()

    f, bounds, f_opt = prepare_benchmark(benchmark)

    # Random baseline
    rand_arg_his, rand_f_his = test_random_sample(f, bounds, n_iter)
    
    # Run models asynchronously
    pool = mp.Pool()
    results_dict = { name: pool.apply_async(method, [f, bounds, n_iter]) for name, method in bo_methods.items() }
    bo_models_tuple = [ (name, result.get()) for name, result in results_dict.items() ]
    bo_names = list(map(itemgetter(0), bo_models_tuple))
    bo_models = list(map(itemgetter(1), bo_models_tuple))
    ir = [acc_ir(bo.model.Y, f_opt) for bo in bo_models]

    ir.insert(0, acc_ir(rand_f_his, f_opt))
    bo_names.insert(0, "random")

    sns.set_style("darkgrid")
    plot_ir(ir)
    plt.legend(bo_names)
    plt.show()

    return bo_models, bo_names, ir


def embed(f, A, f_dim=2):
    """
    Arguments:
        f {function} -- function to embed
        A {np.array} -- The linear scaling for each additional dimension
    
    Keyword Arguments:
        f_dim {int} -- original (default: {2})
    
    Returns:
        {function} -- embedded function with dim `f_dim + A.shape[0]`
    """


    @wraps(f)
    def wrapper(x):
        """
        Arguments:
            x {np.array} -- (n,d) where n is #samples and d is #dimensions.
        
        Returns:
            [type] -- [description]
        """
        return f(x[..., :f_dim]) + A * x[..., f_dim:]
    return wrapper
