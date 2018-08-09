import os
import sys
import argparse
import datetime
import uuid

import pandas
import GPy
import numpy as np
import matplotlib.pyplot as plt
import hpolib.benchmarks.synthetic_functions as hpolib_funcs

from src.tests import prepare_benchmark, plot_ir, acc_ir
from src.bo import BO
from src.acquisition_functions import EI, UCB
from src.bayesian_linear_regression import BayesianLinearRegression, GPyRegression
from src.models import BOModel, GPyBOModel
from src.neural_network import TorchRegressionModel


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model",       type=str,   choices=["dngo", "gp"], default="dngo")

# nn
parser.add_argument("-b", "--dim_basis",     type=int,   default=50)
parser.add_argument("-h1", "--dim_h1",       type=int,   default=50)
parser.add_argument("-h2", "--dim_h2",       type=int,   default=50)
parser.add_argument("-nn", "--num_nn",       type=int,   default=1)
parser.add_argument("-bs", "--batch_size",   type=int,   default=1000) # if none > n_iter
parser.add_argument("-e", "--epochs",        type=int,   default=1000)
parser.add_argument("-lr", "--lr",           type=float, default=0.01)
parser.add_argument("-l2", "--weight_decay", type=float, default=0)

# gp
parser.add_argument("-mcmc", "--num_mcmc", type=int, default=0)

# bo
obj_functions = {
    'branin': hpolib_funcs.Branin,
    'hartmann3': hpolib_funcs.Hartmann3,
    'hartmann6': hpolib_funcs.Hartmann6,
    'camelback': hpolib_funcs.Camelback,
    'forrester': hpolib_funcs.Forrester,
    'bohachevsky': hpolib_funcs.Bohachevsky,
    'goldsteinprice': hpolib_funcs.GoldsteinPrice,
    'levy': hpolib_funcs.Levy,
    'rosenbrock': hpolib_funcs.Rosenbrock,
    'sinone': hpolib_funcs.SinOne,
    'sintwo': hpolib_funcs.SinTwo,
}
obj_functions = {k: prepare_benchmark(Func()) for (k, Func) in obj_functions.items()}
parser.add_argument("-f", "--obj_func", type=str, choices=obj_functions.keys(), default="branin")

acquisition_functions = {'EI': EI,
                         'UCB': UCB,}
parser.add_argument("-a", "--acq", type=str, choices=acquisition_functions.keys(), default="UCB")

parser.add_argument("-k", "--n_iter", type=int, default=100)

args = parser.parse_args()


# Constructing model

f, bounds, f_opt = obj_functions[args.obj_func]
input_dim = bounds.shape[0]

if args.model == "dngo":
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
    model = GPyBOModel(kernel=kernel, num_mcmc=args.num_mcmc, fix_noise=True)
else:
    raise Exception("`dngo` and `gp` should be the only two options")

acq = acquisition_functions[args.acq](model)
bo = BO(f, model, acquisition_function=acq, n_iter=args.n_iter, bounds=bounds)

# outputs/{model_shortname}/
#   command.txt
#   immediate_regret.png
#   obs.npy
#   plots/
#     i-{i}.png

def remove_start_dash(s):
    if s[:2] == "--":
        return s[2:]
    elif s[:1] == "-":
        return s[1:]
    else:
        return s

def get_shorthand(k):
    """Transform e.g. `batch_size` into `bs` for 
    registered argparse argument  `-bs` `--batch_size`
    """
    for action in parser._actions:
        if action.dest == k:
            s = action.option_strings[0]
            return remove_start_dash(s)
    # fallback
    return k

def generate_uuid():
    return str(uuid.uuid4()).split('-')[0]

uid = generate_uuid()
date = datetime.datetime.utcnow()

model_shortname = map(remove_start_dash, sys.argv[1:])
model_shortname = map(get_shorthand, model_shortname)
model_shortname = "-".join(model_shortname)
model_shortname = uid + "--" + model_shortname
# modelname = "-".join([str(get_shorthand(k)) + "=" + str(v) for (k, v) in vars(args).items()])

row = vars(args)
row['uuid'] = uid
row['date'] = date
row['name'] = model_shortname

top_folder = "outputs"
folder = os.path.join(top_folder, model_shortname)
plot_folder = os.path.join(folder, "plots")
os.makedirs(folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

database = os.path.join(top_folder, "entries.csv")
command_path = os.path.join(folder, "command.txt")
obs_X_path = os.path.join(folder, "obs-X.npy")
obs_Y_path = os.path.join(folder, "obs-Y.npy")
regret_plot_path = os.path.join(folder, "immediate_regret.png")

# log command to rerun easily
command = " ".join(sys.argv)
command_full = "python run.py " + " ".join(["--{} {}".format(k, v) for k, v in vars(args).items()])
with open(command_path, 'w') as file:
    file.writelines([command, "\n", command_full])


def backup(bo, i, x_new):
    # Plot step
    path = os.path.join(plot_folder, "i-{}.png".format(i))
    bo.plot_prediction(x_new=x_new, save_dist=path)

    # Update observation record
    np.save(obs_X_path, bo.model.X)
    np.save(obs_Y_path, bo.model.Y)

    # Plot regret
    ir = acc_ir(bo.model.Y, f_opt)
    plot_ir([ir])
    plt.savefig(regret_plot_path)
    plt.close()

    # Update row data
    row['immediate_regret'] = ir[-1]
    row['incumbent'] = bo.model.get_incumbent()[1]
    row['num_steps'] = i

    try:    
        df = pandas.read_csv(database)
        df = df.set_index('uuid')
        df.loc[uid] = pandas.Series(row)
    except FileNotFoundError:
        df = pandas.DataFrame([row])
        df = df.set_index('uuid')
    df.to_csv(database)

bo.run(do_plot=False, periodic_interval=20, periodic_callback=backup)

backup(bo, args.n_iter, None)
