from functools import wraps

import numpy as np
from matplotlib import pyplot as plt


def random_hypercube_samples(n_samples, bounds, rng=None):
    """Random sample from d-dimensional hypercube (d = bounds.shape[0]).

    Returns: (n_samples, dim)
    """

    rng = rng if rng is not None else np.random.RandomState()

    dims = bounds.shape[0]
    a = rng.uniform(0, 1, (dims, n_samples))
    bounds_repeated = np.repeat(bounds[:, :, None], n_samples, axis=2)
    samples = a * np.abs(bounds_repeated[:,1] - bounds_repeated[:,0]) + bounds_repeated[:,0]
    samples = np.swapaxes(samples, 0, 1)

    # This handles the case where the sample is slightly above or below the bounds
    # due to floating point precision.
    return constrain_points(samples, bounds)


def vectorize(f):
    @wraps(f)
    def wrapper(X):
        return np.apply_along_axis(f, -1, X)[..., None]
    return wrapper


def constrain_points(x, bounds):
    dim = x.shape[0]
    minx = np.repeat(bounds[:, 0][None, :], dim, axis=0)
    maxx = np.repeat(bounds[:, 1][None, :], dim, axis=0)
    return np.clip(x, a_min=minx, a_max=maxx)


def immediate_regret(y, y_opt):
    y_opt = y_opt if y_opt is not None else 0
    return np.abs(y - y_opt)


def accumulate_immediate_regret(history, y_opt):
    max_history = np.maximum.accumulate(history)
    regret_history = immediate_regret(max_history, y_opt)
    return regret_history


def plot_immediate_regret(acc_ir_arrays):
    for acc_ir in acc_ir_arrays:
        plt.plot(acc_ir)
    plt.yscale('log')


def prepare_stratified_benchmark(func):
    """"
    Arguments:
        Func -- Function from evalset

    Returns:
        (function, np array of bounds, minimum f value)
    """

    bounds = np.array(func.bounds)
    f_opt = func.fmin

    @wraps(func.evaluate)
    def wrapper(x):
        return -func.evaluate(x)

    return vectorize(wrapper), bounds, f_opt


def prepare_benchmark(func):
    """"
    Arguments:
        Func -- Function from hpolib.benchmarks.synthetic_functions

    Returns:
        (function, np array of bounds, minimum f value)
    """

    info = func.get_meta_information()
    bounds = np.array(info['bounds'])
    f_opt = info.get('f_opt', None)
    if f_opt is not None:
        f_opt = -f_opt

    @wraps(func)
    def wrapper(x):
        return -func(x)

    return vectorize(wrapper), bounds, f_opt