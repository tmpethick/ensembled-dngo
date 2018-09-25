from functools import wraps

import numpy as np


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
        return f(x[..., :f_dim]) + np.sum(A * x[..., f_dim:])
    return wrapper