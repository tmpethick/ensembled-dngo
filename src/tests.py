from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from src.bo import random_hypercube_samples, vectorize

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


def test_dngo_10_10_10_pe(f, bounds, n_iter, do_plot=False):
    from src.bo import BO, random_hypercube_samples, vectorize
    from src.acquisition_functions import EI, UCB
    from src.bayesian_linear_regression import BayesianLinearRegression
    from src.dngo import BOModel
    from src.neural_network import TFModel

    input_dim = bounds.shape[0]
    nn = TFModel(input_dim=input_dim, dim_basis=10, dim_h1=10, dim_h2=10, epochs=1000, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo # bo.model.X, bo.model.Y, f_opt

def test_dngo_50_50_50_pe(f, bounds, n_iter, do_plot=False):

    from src.bo import BO, random_hypercube_samples, vectorize
    from src.acquisition_functions import EI, UCB
    from src.bayesian_linear_regression import BayesianLinearRegression
    from src.dngo import BOModel
    from src.neural_network import TFModel

    input_dim = bounds.shape[0]
    nn = TFModel(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=0)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)
    
    return bo # bo.model.X, bo.model.Y, f_opt

def test_dngo_50_50_50_marg(f, bounds, n_iter, do_plot=False):
    from src.bo import BO, random_hypercube_samples, vectorize
    from src.acquisition_functions import EI, UCB
    from src.bayesian_linear_regression import BayesianLinearRegression
    from src.dngo import BOModel
    from src.neural_network import TFModel

    input_dim = bounds.shape[0]
    nn = TFModel(input_dim=input_dim, dim_basis=50, epochs=1000, batch_size=10)
    reg = BayesianLinearRegression(num_mcmc=20, burn_in=1000, mcmc_steps=1000)
    model = BOModel(nn, regressor=reg)
    # acq = EI(model, par=0.01)
    acq = UCB(model)
    bo = BO(f, model, acquisition_function=acq, n_iter=n_iter, bounds=bounds)
    bo.run(do_plot=do_plot)

    return bo # bo.model.X, bo.model.Y, f_opt
