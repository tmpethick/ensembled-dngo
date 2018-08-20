#!/usr/bin/env python
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
plt.switch_backend('agg')

from src.tests import prepare_benchmark, plot_ir, acc_ir, embed
from src.bo import BO
from src.acquisition_functions import EI, UCB
from src.bayesian_linear_regression import GPyRegression
from src.models import BOModel, GPyBOModel
from src.neural_network import TorchRegressionModel
from src.logistic_regression_benchmark import LogisticRegression
from src.rosenbrock_benchmark import Rosenbrock
import config

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--seed",  type=int, default=None)
parser.add_argument("-m", "--model", type=str, default="dngo", choices=["dngo", "gp"])

# nn
parser.add_argument("-b", "--dim_basis",     type=int,   default=50)
parser.add_argument("-h1", "--dim_h1",       type=int,   default=50)
parser.add_argument("-h2", "--dim_h2",       type=int,   default=50)
parser.add_argument("-nn", "--num_nn",       type=int,   default=1)
parser.add_argument("-bs", "--batch_size",   type=int,   default=1000) # if none > n_iter
parser.add_argument("-e", "--epochs",        type=int,   default=1000)
parser.add_argument("-lr", "--lr",           type=float, default=0.01)
parser.add_argument("-l2", "--weight_decay", type=float, default=0)
nn_aggregators = {
    "median": np.median,
    "max": np.max,
    "average": np.average,
}
parser.add_argument("-agg", "--nn_aggregator", type=str, default="median", choices=nn_aggregators.keys())

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
obj_functions['logistic_regression_mnist'] = prepare_benchmark(LogisticRegression(num_epochs=10))
obj_functions['rosenbrock10D'] = prepare_benchmark(Rosenbrock(d=10))
parser.add_argument("-f", "--obj_func", type=str, choices=obj_functions.keys(), default="branin")

parser.add_argument("-em", "--embedding", nargs='+', type=float, default=None)

acquisition_functions = {'EI': EI,
                         'UCB': UCB,}
parser.add_argument("-a", "--acq", type=str, choices=acquisition_functions.keys(), default="UCB")

parser.add_argument("-k",    "--n_iter", type=int, default=100)
parser.add_argument("-init", "--n_init", type=int, default=2)


# Constructing model

def create_model(args):
    args_dict = vars(args)
    seed = args_dict.get('seed', None)
    rng = np.random.RandomState(seed)
    
    # Support function parsed directly
    if type(args.obj_func) == tuple:
        f, bounds, f_opt = args.obj_func
    else:
        f, bounds, f_opt = obj_functions[args.obj_func]

    input_dim = bounds.shape[0]

    # Embedding
    embedding = args_dict.get('embedding')
    if type(embedding) is list and len(embedding) > 0:
        embedded_dims = len(embedding)
        A = np.array(embedding)
        bounds = np.concatenate([bounds, [[0,1]] * embedded_dims])
        f = embed(f, A, f_dim=input_dim)
        input_dim = bounds.shape[0]
    else:
        embedded_dims = None

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
        kernel = GPy.kern.Linear(args.dim_basis)
        kernel.variances.set_prior(GPy.priors.LogGaussian(0, 1))
        reg = GPyRegression(kernel=kernel, num_mcmc=args.num_mcmc, fix_noise=True)
        # reg = BayesianLinearRegression(num_mcmc=args.num_mcmc)
        model = BOModel(nn, regressor=reg, num_nn=args.num_nn)
    elif args.model == "gp":
        kernel = GPy.kern.RBF(input_dim)
        kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
        model = GPyBOModel(kernel=kernel, num_mcmc=args.num_mcmc, fix_noise=True)
    else:
        raise Exception("`dngo` and `gp` should be the only two options")

    acq = acquisition_functions[args.acq](model)
    bo = BO(f, model, 
        acquisition_function=acq, 
        n_init=args_dict.get('n_init', 2), 
        n_iter=args.n_iter, 
        bounds=bounds, 
        embedded_dims=embedded_dims,
        f_opt=f_opt, 
        rng=rng)
    return bo

if __name__ == '__main__':
    args = parser.parse_args()
    bo = create_model(args)

    # outputs/{model_shortname}/
    #   command.txt
    #   immediate_regret.png
    #   obs.npy
    #   plots/
    #     i-{i}.png
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

    def remove_start_dash(s):
        if s[:2] == "--":
            return s[2:]
        elif s[:1] == "-":
            return s[1:]
        else:
            return s

    uid = generate_uuid()
    date = datetime.datetime.utcnow()

    model_shortname = map(remove_start_dash, sys.argv[1:])
    model_shortname = map(get_shorthand, model_shortname)
    model_shortname = "-".join(model_shortname)
    model_shortname = uid + "--" + model_shortname

    conf = config.get_config()
    model_conf = config.get_model_config(uid, model_shortname, conf)

    row = vars(args)
    row['uuid'] = uid
    row['date'] = date
    row['name'] = model_shortname

    os.makedirs(model_conf['folder'], exist_ok=True)
    os.makedirs(model_conf['plot_folder'], exist_ok=True)

    # log command to rerun easily
    command = " ".join(sys.argv)
    command_full = "python run.py " + " ".join(["--{} {}".format(k, v) for k, v in vars(args).items()])
    with open(model_conf['command_path'], 'w') as file:
        file.writelines([command, "\n", command_full])

    has_quick_eval = args.obj_func is not 'logistic_regression_mnist'

    def backup(bo, i, x_new):
        if has_quick_eval:
            # Plot step
            path = os.path.join(model_conf['plot_folder'], "i-{}.png".format(i))
            bo.plot_prediction(x_new=x_new, save_dist=path)
            
            if args.embedding is not None:
                path = os.path.join(model_conf['plot_folder'], "i-embedding-{}.png".format(i))
                bo.plot_prediction(x_new=x_new, save_dist=path, plot_embedded_subspace=True)

            # Plot regret
            ir = acc_ir(bo.model.Y, bo.f_opt)
            plot_ir([ir])
            plt.savefig(model_conf['regret_plot_path'])
            plt.close()

        # Update observation record
        np.save(model_conf['obs_X_path'], bo.model.X)
        np.save(model_conf['obs_Y_path'], bo.model.Y)

        # Update row data
        row['immediate_regret'] = ir[-1]
        row['incumbent'] = bo.model.get_incumbent()[1]
        row['num_steps'] = i

        try:
            df = pandas.read_csv(conf['database'])

            # Add missing columns (in case a new parameter was introduced)
            missing_columns = set(row.keys()).symmetric_difference(set(df.columns.values))
            for col in missing_columns:
                df[col] = pandas.Series(None, index=df.index)

            df = df.set_index('uuid')
            df.loc[uid] = pandas.Series(row)
            df = df.reset_index()
        except FileNotFoundError:
            df = pandas.DataFrame([row])
        df.to_csv(conf['database'], index=False)

    bo.run(do_plot=False, periodic_interval=10, periodic_callback=backup)

    backup(bo, args.n_iter, None)
