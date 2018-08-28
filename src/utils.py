from functools import wraps

import numpy as np


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

    # lower = bounds[:,0]
    # upper = bounds[:,1]
    # if rng is None:
    #     rng = np.random.RandomState(np.random.randint(0, 10000))
    # n_dims = lower.shape[0]
    # # Generate bounds for random number generator
    # s_bounds = np.array([np.linspace(lower[i], upper[i], n_points + 1) for i in range(n_dims)])
    # s_lower = s_bounds[:, :-1]
    # s_upper = s_bounds[:, 1:]
    # # Generate samples
    # samples = s_lower + rng.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # # Shuffle samples in each dimension
    # for i in range(n_dims):
    #     rng.shuffle(samples[i, :])
    # return samples.T


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
