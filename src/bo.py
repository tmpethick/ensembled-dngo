import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.utils import random_hypercube_samples
from .priors import *
from .acquisition_functions import UCB
from .tests import acc_ir, plot_ir


class BO(object):
    def __init__(self, obj_func, model, acquisition_function=None, n_iter = 10, bounds=np.array([[0,1]]), f_opt=None):
        self.n_iter = n_iter
        self.bounds = bounds
        self.obj_func = obj_func
        self.model = model
        self.f_opt = f_opt

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

    def plot_prediction(self, x_new=None, bounds=None, save_dist=None):
        if bounds is None:  
            bounds = self.bounds
        dims = self.bounds.shape[0]
        if dims == 2:
            x0 = np.linspace(bounds[0,0], bounds[0,1], 100)
            x1 = np.linspace(bounds[1,0], bounds[1,1], 100)
            x0v, x1v = np.meshgrid(x0, x1)
            xinput = np.swapaxes(np.array([x0v, x1v]), 0, -1) # set dim axis to last
            xinput = np.swapaxes(xinput, 0, 1)                # swap x0 and x1 axis
            origin_shape = xinput.shape[:-1]
            flattenxinput = xinput.reshape(-1, dims)
            acq = self.model.acq(flattenxinput, self.acquisition_function.calc)
            acq = np.reshape(acq, origin_shape)

            # (ensemble, hyperparams, summarystats, samples)
            summ = self.model.predict(flattenxinput)
            if len(summ.shape) == 4:
                mean = summ[:, :, 0, :]
                mean = np.average(mean, axis=(0, 1))
            else:
                mean = summ[0]
            mean = np.reshape(mean, origin_shape)
            
            y = self.obj_func(xinput)[..., 0]
            
            X = self.model.X
            idx = (bounds[0,0] < X[:,0]) & (X[:,0] < bounds[0,1]) & (bounds[1,0] < X[:,1]) & (X[:,1] < bounds[1,1])
            X0 = X[idx,0]
            X1 = X[idx,1]

            # Plot acq
            plt.subplot(1, 3, 1)
            plt.contourf(x0v,x1v, acq, 24)
            plt.scatter(X0, X1)

            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

            # Plot f
            plt.subplot(1, 3, 2)
            plt.contourf(x0v,x1v, y, 24)
            plt.scatter(X0, X1)
            
            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

            # Plot mean
            plt.subplot(1, 3, 3)
            plt.contourf(x0v,x1v, mean, 24)
            plt.scatter(X0, X1)
            
            if x_new is not None:
                plt.plot([x_new[0]], [x_new[1]], marker='x', markersize=20, color="white")

        elif dims == 1:
            X_line = np.linspace(bounds[:, 0], bounds[:, 1], 100)[:, None]
            Y_line = self.obj_func(X_line)

            plt.subplot(1, 2, 1)
            self.model.plot_prediction(X_line, Y_line, x_new=x_new)
            plt.subplot(1, 2, 2)
            self.model.plot_acq(X_line, self.acquisition_function.calc)

        if save_dist is not None:
            plt.savefig(save_dist)
            plt.close()
        else:
            plt.show()

    def run(self, n_kickstart=2, do_plot=True, periodic_interval=20, periodic_callback=None):
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

            if periodic_callback is not None \
                and i is not 0 \
                and i % periodic_interval == 0:
                    periodic_callback(self, i, x_new)

            if do_plot:
                self.plot_prediction(x_new=x_new)

                if i is not 0 and i % 20 == 0:
                    if self.f_opt is not None:
                        ir = acc_ir(self.model.Y, self.f_opt)
                        plot_ir([ir])
                        plt.show()
                    else: 
                        warnings.warn("`f_opt` is not provided, so plotting regret is impossible.")

            self.model.add_observations(X_new, Y_new)

        if do_plot:
            self.plot_prediction()
