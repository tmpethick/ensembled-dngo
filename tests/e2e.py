from operator import itemgetter

import GPy
import matplotlib.pyplot as plt
import seaborn as sns
import pathos.multiprocessing as mp
# import torch.multiprocessing as mp
from hpolib.benchmarks.synthetic_functions import Branin

from src.utils import random_hypercube_samples, accumulate_immediate_regret, plot_immediate_regret, prepare_benchmark


def test_random_sample(f, bounds, n_iter=100):
    # Random sample
    R_samples = random_hypercube_samples(n_iter, bounds)
    R_values = f(R_samples)
    # R_values[np.argmax(R_values)]
    return R_samples, R_values


def test_gp(f, bounds, n_iter, do_plot=False):
    import GPy
    from src.models.models import GPModel
    from .acquisition_functions import UCB
    from .algorithm import AcquisitionAlgorithm

    input_dim = bounds.shape[0]

    kernel = GPy.kern.RBF(input_dim)
    kernel.variance.set_prior(GPy.priors.LogGaussian(0.005, 0.5)) # log_prior()
    model = GPModel(kernel=kernel, num_mcmc=0, fix_noise=True)

    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    return bo

def test_dngo_10_10_10_marg(f, bounds, n_iter, do_plot=False):
    from .algorithm import AcquisitionAlgorithm
    from .acquisition_functions import UCB
    from src.models.regression import BayesianLinearRegression
    from src.models.models import DNGOModel
    from src.models.neural_network import FeatureExtractorNetwork

    input_dim = bounds.shape[0]
    nn = FeatureExtractorNetwork(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=6)
    model = DNGOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo

def test_dngo_10_10_10_pe_ensemble(f, bounds, n_iter, do_plot=False):
    from .algorithm import AcquisitionAlgorithm
    from .acquisition_functions import UCB
    from src.models.regression import BayesianLinearRegression
    from src.models.models import DNGOModel
    from src.models.neural_network import FeatureExtractorNetwork

    input_dim = bounds.shape[0]
    nn = FeatureExtractorNetwork(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = DNGOModel(nn, regressor=reg, num_nn=5)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo


def test_dngo_10_10_10_pe(f, bounds, n_iter, do_plot=False):
    from .algorithm import AcquisitionAlgorithm
    from .acquisition_functions import UCB
    from src.models.regression import BayesianLinearRegression
    from src.models.models import DNGOModel
    from src.models.neural_network import FeatureExtractorNetwork

    input_dim = bounds.shape[0]
    nn = FeatureExtractorNetwork(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = DNGOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo

def test_dngo_50_50_50_pe(f, bounds, n_iter, do_plot=False):

    from .algorithm import AcquisitionAlgorithm
    from .acquisition_functions import UCB
    from src.models.regression import BayesianLinearRegression
    from src.models.models import DNGOModel
    from src.models.neural_network import FeatureExtractorNetwork

    input_dim = bounds.shape[0]
    nn = FeatureExtractorNetwork(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = DNGOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo


def test_dngo_50_50_50_marg(f, bounds, n_iter, do_plot=False):
    from .algorithm import AcquisitionAlgorithm
    from .acquisition_functions import UCB
    from src.models.regression import BayesianLinearRegression
    from src.models.models import DNGOModel
    from src.models.neural_network import FeatureExtractorNetwork

    input_dim = bounds.shape[0]
    nn = FeatureExtractorNetwork(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=1000)
    reg = BayesianLinearRegression(num_mcmc=20, burn_in=1000, mcmc_steps=1000)
    model = DNGOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = AcquisitionAlgorithm(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)

    return bo

def test_method(bo_method, n_iter=200, Benchmark=None, do_plot=False):
    benchmark = Benchmark() if Benchmark is not None else Branin()
    f, bounds, f_opt = prepare_benchmark(benchmark)
    bo = bo_method(f, bounds, n_iter, do_plot=do_plot)
    ir = accumulate_immediate_regret(bo.model.Y, f_opt)

    sns.set_style("darkgrid")
    plot_immediate_regret([ir])

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
    ir = [accumulate_immediate_regret(bo.model.Y, f_opt) for bo in bo_models]

    ir.insert(0, accumulate_immediate_regret(rand_f_his, f_opt))
    bo_names.insert(0, "random")

    sns.set_style("darkgrid")
    plot_immediate_regret(ir)
    plt.legend(bo_names)
    plt.show()

    return bo_models, bo_names, ir


