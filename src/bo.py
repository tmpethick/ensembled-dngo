from functools import wraps

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .priors import *
from .acquisition_functions import UCB


def vectorize(f):
    @wraps(f)
    def wrapper(X):
        return np.apply_along_axis(f, -1, X)[..., None]
    return wrapper


def random_hypercube_samples(n_samples, bounds):
    """Random sample from d-dimensional hypercube (d = bounds.shape[0]).

    Returns: (n_samples, dim)
    """

    dims = bounds.shape[0]
    a = np.random.uniform(0, 1, (dims, n_samples))
    bounds_repeated = np.repeat(bounds[:, :, None], n_samples, axis=2)
    samples = a * np.abs(bounds_repeated[:,1] - bounds_repeated[:,0]) + bounds_repeated[:,0]
    return np.swapaxes(samples, 0, 1)


def constrain_point(x, bounds):
    minx = bounds[:, 0]
    maxx = bounds[:, 1]
    x = np.maximum(x, minx)
    return np.minimum(x, maxx)


class BO(object):
    def __init__(self, obj_func, model, acquisition_function=None, n_iter = 10, bounds=np.array([[0,1]])):
        self.n_iter = n_iter
        self.bounds = bounds
        self.obj_func = obj_func
        self.model = model

        if acquisition_function is None:
            self.acquisition_function = UCB(self.model)
        else:
            self.acquisition_function = acquisition_function

    def max_acq(self, n_starts=100):        
        min_y = float("inf")
        min_x = None

        def min_obj(x):
            """lift into array and negate.
            """
            x = np.array([x])
            return -self.model.acq(x, self.acquisition_function.calc)[0]

        # TODO: turn into numpy operations to parallelize
        for x0 in random_hypercube_samples(n_starts, self.bounds):
            # This handles the case where the sample is slightly above or below the bounds
            # due to floating point precision.
            x0 = constrain_point(x0, self.bounds)
            
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 

        return min_x

    def plot_2D_surface(self, use_plotly=False):
        dims = self.bounds.shape[0]

        if dims != 2:
            raise Exception("Input has to be 2-dimensional")

        x0 = np.linspace(self.bounds[0,0], self.bounds[0,1], 50)
        x1 = np.linspace(self.bounds[1,0], self.bounds[1,1], 100)
        x0v, x1v = np.meshgrid(x0, x1)
        xinput = np.swapaxes(np.array([x0v, x1v]), 0, -1) # set dim axis to last
        xinput = np.swapaxes(xinput, 0, 1)                # swap x0 and x1 axis

        # Calc prediction on grid
        origin_shape = xinput.shape[:-1]
        flattenxinput = xinput.reshape(-1, dims)
        mean, var = self.model.predict(flattenxinput)
        mean = np.reshape(mean, origin_shape)
        var = np.reshape(var, origin_shape)
        acq = self.model.acq(flattenxinput, self.acquisition_function.calc)
        acq = np.reshape(acq, origin_shape)

        y = self.obj_func(xinput)[..., 0]

        if use_plotly:
            import plotly.tools as tls
            import plotly.graph_objs as go

            layout = dict(
                width=1000,
                height=1000,
                autosize=False,
                margin=dict(t=0, b=0, r=0, l=0),
            )
            fig = tls.make_subplots(rows=2, cols=2, specs=[[{'is_3d': True}] * 2] * 2)
            fig['layout'].update(**layout)

            scatter = go.Scatter3d(x=self.model.X[:,0], y=self.model.X[:,1], z=self.model.Y[:,0], mode = 'markers')            
            surface1 = go.Surface(x=x0v, y=x1v, z=mean)
            surface2 = go.Surface(x=x0v, y=x1v, z=mean + np.sqrt(var), opacity=0.5)
            fig.append_trace(scatter, 1, 1)
            fig.append_trace(surface1, 1, 1)
            fig.append_trace(surface2, 1, 1)

            surface1 = go.Surface(x=x0v, y=x1v, z=y)
            fig.append_trace(scatter, 1, 2)
            fig.append_trace(surface1, 1, 2)

            surface1 = go.Surface(x=x0v, y=x1v, z=acq)
            fig.append_trace(surface1, 2, 1)

            return fig
        else:
            # Plot prediction
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(self.model.X[:,0], self.model.X[:,1], self.model.Y[:,0], color="red")
            ax.plot_surface(x0v, x1v, mean)
            ax.plot_surface(x0v, x1v, mean + np.sqrt(var), alpha=0.5)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(self.model.X[:,0], self.model.X[:,1], self.model.Y[:,0], color="red")
            ax.plot_surface(x0v, x1v, y)
            return fig

    def plot_prediction(self, x_new=None):
        dims = self.bounds.shape[0]
        if dims == 2:
            x0 = np.linspace(self.bounds[0,0], self.bounds[0,1], 100)
            x1 = np.linspace(self.bounds[1,0], self.bounds[1,1], 100)
            x0v, x1v = np.meshgrid(x0, x1)
            xinput = np.swapaxes(np.array([x0v, x1v]), 0, -1) # set dim axis to last
            xinput = np.swapaxes(xinput, 0, 1)                # swap x0 and x1 axis
            origin_shape = xinput.shape[:-1]
            flattenxinput = xinput.reshape(-1, dims)
            acq = self.model.acq(flattenxinput, self.acquisition_function.calc)
            acq = np.reshape(acq, origin_shape)
            
            y = self.obj_func(xinput)[..., 0]
            
            # Plot acq
            plt.subplot(1, 2, 1)
            plt.contourf(x0v,x1v, acq, 24)
            plt.scatter(self.model.X[:,0], self.model.X[:,1])

            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

            # Plot f
            plt.subplot(1, 2, 2)
            plt.contourf(x0v,x1v, y, 24)
            plt.scatter(self.model.X[:,0], self.model.X[:,1])
            
            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")
            
            plt.show()

        elif dims == 1:
            X_line = np.linspace(self.bounds[:, 0], self.bounds[:, 1], 100)[:, None]
            Y_line = self.obj_func(X_line)

            plt.subplot(1, 2, 1)
            self.model.plot_prediction(X_line, Y_line, x_new=x_new)
            plt.subplot(1, 2, 2)
            self.model.plot_acq(X_line, self.acquisition_function.calc)
            plt.show()

    def run(self, n_kickstart=2, do_plot=True):
        # Data
        X = random_hypercube_samples(n_kickstart, self.bounds)
        Y = self.obj_func(X)
        self.model.init(X,Y)

        for i in range(0, self.n_iter):
            print("... starting round", i, "/", self.n_iter)

            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            Y_new = self.obj_func(X_new)

            if do_plot:
                self.plot_prediction(x_new=x_new)

            self.model.add_observations(X_new, Y_new)

        if do_plot:
            self.plot_prediction()

        # TODO: Move session outside..
        if hasattr(self.model, "sess"):
            self.model.sess.close()
            self.model.sess = None
