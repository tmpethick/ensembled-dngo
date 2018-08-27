import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.utils import random_hypercube_samples, constrain_points
from .priors import *
from .acquisition_functions import UCB
from .tests import acc_ir, plot_ir
from src.models import DummyModel


class BO(object):
    def __init__(self, obj_func, model, acquisition_function=None, n_init=20, n_iter = 10, bounds=np.array([[0,1]]), f_opt=None, rng=None, embedded_dims=None):
        self.n_iter = n_iter
        self.n_init = n_init
        self.bounds = bounds
        self.obj_func = obj_func
        self.model = model
        self.f_opt = f_opt
        self.embedded_dims = embedded_dims

        # Only used in random sample so far.
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        if acquisition_function is None:
            self.acquisition_function = UCB(self.model)
        else:
            self.acquisition_function = acquisition_function

    def max_acq(self, n_starts=200):
        min_y = float("inf")
        min_x = None

        def min_obj(x):
            """lift into array and negate.
            """
            x = np.array([x])
            return -self.model.acq(x, self.acquisition_function.calc)[0]

        # TODO: turn into numpy operations to parallelize
        for x0 in random_hypercube_samples(n_starts, self.bounds, rng=self.rng):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B') 
            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 
            elif min_x is None: # fix if no point is <inf
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
        # xinput -- shape: (n,d)

        # Calc prediction on grid
        origin_shape = xinput.shape[:-1]
        flattenxinput = xinput.reshape(-1, dims)
        predictions = self.model.predict(flattenxinput)

        # Collapse hyperparam and ensemble dimensions
        predictions = predictions.reshape(-1, 2, xinput.shape[0], xinput.shape[1])

        # Plot one of the means + vars
        summ = predictions[0]
        mean = np.reshape(summ[0], origin_shape)
        var = np.reshape(summ[1], origin_shape)

        acq = self.model.acq(flattenxinput, self.acquisition_function.calc)
        acq = np.reshape(acq, origin_shape)

        y = self.obj_func(flattenxinput)[..., 0]
        y = np.reshape(y, origin_shape)

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

    def plot_prediction(self, x_new=None, bounds=None, save_dist=None, plot_predictions=True, plot_embedded_subspace=False):
        if bounds is None:  
            bounds = self.bounds
        dims = self.bounds.shape[0]

        if plot_embedded_subspace and self.embedded_dims is not None:
            embedded_dims = dims - self.embedded_dims
        else:
            embedded_dims = dims
        
        if embedded_dims == 2:
            x0 = np.linspace(bounds[0,0], bounds[0,1], 100)
            x1 = np.linspace(bounds[1,0], bounds[1,1], 100)
            x0v, x1v = np.meshgrid(x0, x1)
            xinput = np.swapaxes(np.array([x0v, x1v]), 0, -1) # set dim axis to last
            xinput = np.swapaxes(xinput, 0, 1)                # swap x0 and x1 axis

            # expand the 2D xinput to embedded space.
            # TODO: make 1 configurable
            if plot_embedded_subspace and self.embedded_dims is not None:
                xinput = np.append(xinput, np.ones(xinput.shape[:-1] + (self.embedded_dims,)), axis=-1)
            
            origin_shape = xinput.shape[:-1]
            flattenxinput = xinput.reshape(-1, dims)
                        
            X = self.model.X
            idx = (bounds[0,0] <= X[:,0]) & (X[:,0] <= bounds[0,1]) & (bounds[1,0] <= X[:,1]) & (X[:,1] <= bounds[1,1])
            X0 = X[idx,0]
            X1 = X[idx,1]

            if plot_predictions:
                acq = self.model.acq(flattenxinput, self.acquisition_function.calc)
                acq = np.reshape(acq, origin_shape)

                # Plot acq
                plt.subplot(1, 3, 1)
                plt.contourf(x0v,x1v, acq, 24)
                plt.scatter(X0, X1)

                if x_new is not None:
                    plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

            y = self.obj_func(flattenxinput)[..., 0]
            y = np.reshape(y, origin_shape)

            # Plot f
            if plot_predictions:
                plt.subplot(1, 3, 2)

            plt.contourf(x0v,x1v, y, 24)
            plt.scatter(X0, X1)
            
            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

            if plot_predictions:
                # (ensemble, hyperparams, summarystats, samples)
                summ = self.model.predict(flattenxinput)
                if len(summ.shape) == 4:
                    mean = summ[:, :, 0, :]
                    mean = np.average(mean, axis=(0, 1))
                else:
                    mean = summ[0]
                mean = np.reshape(mean, origin_shape)

                # Plot mean
                if plot_predictions:
                    plt.subplot(1, 3, 3)
                
                plt.contourf(x0v,x1v, mean, 24)
                plt.scatter(X0, X1)
                
                if x_new is not None:
                    plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

        elif embedded_dims == 1:
            X_line = np.linspace(bounds[0, 0], bounds[0, 1], 100)[:, None]
            if plot_embedded_subspace and self.embedded_dims is not None:
                X_line_embedded = np.append(X_line, np.ones(X_line.shape[:-1] + (self.embedded_dims,)), axis=-1)
            else:
                X_line_embedded = X_line

            Y_line = self.obj_func(X_line_embedded)

            plt.subplot(1, 2, 1)
            self.model.plot_prediction(X_line, Y_line, X_embedding=X_line_embedded, x_new=x_new, plot_predictions=plot_predictions)

            if plot_predictions:
                plt.subplot(1, 2, 2)
                self.model.plot_acq(X_line_embedded, self.acquisition_function.calc)
        
        else:
            # Only plot or save if a figure has been constructed.
            return None

        if save_dist is not None:
            plt.savefig(save_dist)
            plt.close()
        else:
            plt.show()

    def run(self, do_plot=True, periodic_interval=20, periodic_callback=None):
        # Data
        X = random_hypercube_samples(self.n_init, self.bounds, rng=self.rng)
        Y = self.obj_func(X)
        self.model.init(X,Y)

        for i in range(0, self.n_iter):
            print("... starting round", i, "/", self.n_iter)

            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            X_new = constrain_points(X_new, self.bounds)
            Y_new = self.obj_func(X_new)

            if periodic_callback is not None \
                and i is not 0 \
                and i % periodic_interval == 0:
                    periodic_callback(self, i, x_new)

            if do_plot:
                self.plot_prediction(x_new=x_new)

                if i is not 0 and i % 20 == 0:
                    ir = acc_ir(self.model.Y, self.f_opt)
                    plot_ir([ir])
                    plt.show()

            self.model.add_observations(X_new, Y_new)

        if do_plot:
            self.plot_prediction()


class RandomSearch(object):
    def __init__(self, obj_func, n_iter = 10, bounds=np.array([[0,1]]), f_opt=None, rng=None):
        self.i = 0
        self.samples = random_hypercube_samples(n_iter, bounds, rng=rng)
        self.bounds = bounds
        self.n_iter = n_iter
        self.obj_func = obj_func
        self.f_opt = f_opt
        self.model = DummyModel()
       
    def max_acq(self, n_starts=200):
        return self.samples[self.i]
       
    def plot_2D_surface(self, use_plotly=False):
        pass
       
    def plot_prediction(self, x_new=None, bounds=None, save_dist=None, plot_predictions=True, plot_embedded_subspace=False):
        pass

    def run(self, do_plot=True, periodic_interval=20, periodic_callback=None):
        self.model.init(np.empty((0,self.bounds.shape[0])), np.empty((0,1)))

        for i in range(0, self.n_iter):
            print("... starting round", i, "/", self.n_iter)
            self.i = i
            
            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            X_new = constrain_points(X_new, self.bounds)
            Y_new = self.obj_func(X_new)

            if periodic_callback is not None \
                and i is not 0 \
                and i % periodic_interval == 0:
                    periodic_callback(self, i, x_new)

            self.model.add_observations(X_new, Y_new)
