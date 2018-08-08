import os
import sys
import argparse

import GPy
import numpy as np
import matplotlib.pyplot as plt
from hpolib.benchmarks.synthetic_functions import Camelback, Branin

from src.tests import prepare_benchmark, plot_ir, acc_ir
from src.acquisition_functions import EI, UCB
from src.bo import BO
from src.acquisition_functions import EI, UCB
from src.bayesian_linear_regression import BayesianLinearRegression, GPyRegression
from src.models import BOModel, GPyBOModel
from src.neural_network import TorchRegressionModel


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model",       type=str,   choices=["dngo", "gp"], default="dngo")

# nn
parser.add_argument("-b", "--dim_basis",     type=int,   default=10)
parser.add_argument("-h1", "--dim_h1",       type=int,   default=10)
parser.add_argument("-h2", "--dim_h2",       type=int,   default=10)
parser.add_argument("-nn", "--num_nn",       type=int,   default=1)
parser.add_argument("-bs", "--batch_size",   type=int,   default=1000) # if none > n_iter
parser.add_argument("-e", "--epochs",        type=int,   default=1000)
parser.add_argument("-lr", "--lr",           type=float, default=0.01)
parser.add_argument("-l2", "--weight_decay", type=int,   default=0)

# gp
parser.add_argument("-mcmc", "--num_mcmc", type=int, default=0)

# bo
obj_functions = {'branin': Branin,
                 'camelback': Camelback,}
obj_functions = {k: prepare_benchmark(Func()) for (k, Func) in obj_functions.items()}
parser.add_argument("-f", "--obj_func", type=str, choices=obj_functions.keys(), default="branin")

acquisition_functions = {'EI': EI,
                         'UCB': UCB,}
parser.add_argument("-a", "--acq", type=str, choices=acquisition_functions.keys(), default="UCB")

parser.add_argument("-k", "--n_iter", type=int, default=100)

args = parser.parse_args()


# Constructing model

f, bounds, f_opt = obj_functions[args.obj_func]

if args.model == "dngo":
    input_dim = bounds.shape[0]
    nn = TorchRegressionModel(
        input_dim=input_dim, 
        dim_basis=args.dim_basis, 
        dim_h1=args.dim_h1, 
        dim_h2=args.dim_h2, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        weight_decay=args.weight_decay)
    # reg = BayesianLinearRegression(num_mcmc=num_mcmc)
    kernel = GPy.kern.Linear(args.dim_basis)
    kernel.variances.set_prior(GPy.priors.LogGaussian(0, 1))
    reg = GPyRegression(kernel=kernel, num_mcmc=args.num_mcmc, fix_noise=True)

    model = BOModel(nn, regressor=reg, num_nn=args.num_nn)
elif args.model == "gp":
    kernel = GPy.kern.RBF(input_dim)
    kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
    model = GPyBOModel(kernel=kernel, num_mcmc=0, fix_noise=True)
else:
    raise Exception("`dngo` and `gp` should be the only two options")

acq = acquisition_functions[args.acq](model)
bo = BO(f, model, acquisition_function=acq, n_iter=args.n_iter, bounds=bounds)

# outputs/modelname/
#   command.txt
#   immediate_regret.png
#   obs.npy
#   plots/
#     i-{i}-{name}.png

def get_shorthand(k):
    """Transform e.g. `batch_size` into `bs` for 
    registered argparse argument  `-bs` `--batch_size`
    """
    for action in parser._actions:
        if action.dest == k:
            s = action.option_strings[0]
            if s[:1] == "-":
                return s[1:]
            elif s[:2] == "--":
                return s[2:]

modelname = "-".join([str(get_shorthand(k)) + "=" + str(v) for (k, v) in vars(args).items()])
folder = os.path.join("outputs", modelname)
plot_folder = os.path.join(folder, "plots")
os.makedirs(folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

command_path = os.path.join(folder, "command.txt")
obs_X_path = os.path.join(folder, "obs-X.npy")
obs_Y_path = os.path.join(folder, "obs-Y.npy")
regret_plot_path = os.path.join(folder, "immediate_regret.png")

# log command to rerun easily
command = " ".join(sys.argv)
command_full = "python run.py " + " ".join(["--{} {}".format(k, v) for k, v in vars(args).items()])
with open(command_path, 'w') as file:
    file.writelines([command, command_full])

def backup(bo, i, x_new):
    # Plot step
    path = os.path.join(plot_folder, "i-{}-{}.png".format(i, modelname))
    bo.plot_prediction(x_new=x_new, save_dist=path)

    # Update observation record
    np.save(obs_X_path, bo.model.X)
    np.save(obs_Y_path, bo.model.Y)

    # Plot regret
    plot_ir([acc_ir(bo.model.Y, f_opt)])
    plt.savefig(regret_plot_path)
    plt.close()

bo.run(do_plot=False, periodic_interval=20, periodic_callback=backup)

backup(bo)
