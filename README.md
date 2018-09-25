# Ensembled Deep Network for Global Optimization

This includes the code and results for the *Ensembled Deep Network for Global Optimization* with the purpose of reproducibility.
For a proper introduction to the model and results please see the [accompanying write-up](paper/main.pdf).

Note that in order to create the results several days were used across ca. 50 CPUs.
For convenience the regret history together with the associated samples have been included so that the behavior of models can be explored through a Jupyter notebook.

Most of the library is self-contained (apart from the usual dependencies like NumPy and PyTorch) -- everything from the bayesian linear regressor to the bayesian optimization procedure is implemented from scratch.
The only two sub-routines of the computational heavy tasks that have been outsourced to external libraries are the MCMC implementation and the cholesky decomposition.

The repository consist of:

- **[a Jupyter Notebook](notebook.ipynb)**: to recreate paper plots, explore results from experiments, run new experiments interactively.
- **[HPC](#executing-on-server)**: a workflow for running experiments in parallel on a HPC that uses the IBM LSF batch system.
- **[Paper](paper/main.pdf)**: which describes the model and results.
- **[RoBO and Spearmint](#comparisons)**: a docker instance and script, respectively, to run benchmark models.


## Code Outline

The code (i.e. everything in `src/`) is roughly divided into four parts:

- BO
- Models
  - Linear Bayesian Regression
  - Deep Neural Network
  - DNGO
  - Ensemble DNGO model
- Benchmark
  - Embedding
  - Hyperparameter optimization of Logistic Regression
- Priors


## Explore Results (Jupyter Notebook)

**Note**: First consolidate [the installation steps](#installation) before attempting to use the notebook.

The notebook found at [`./notebook.ipynb`](notebook.ipynb) served three purposes:

- Recreation of the plots from the write-up.
- Exploration the acquisition landscape and regret plot for a given experiment from the write-up.
- Running a new model programmatically. <br>
  (every configuration is done through a shared interface with `run.py` to ensure reproductivity and allow for calculating a confidence interval based on the aggregated result of identical model configurations).


## Installation

- Conda requirement for server:
  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash miniconda.sh
  ```

- General requirements:
  ```bash
  conda create -n eth python=3.6
  source $HOME/miniconda/bin/activate
  source activate eth
  conda install -y pytorch-cpu torchvision-cpu -c pytorch # gpu: conda install pytorch torchvision -c pytorch
  conda install -y -c conda-forge blas seaborn scipy matplotlib pandas gpy pathos emcee
  pip install pydot-ng
  git clone https://github.com/automl/HPOlib2.git
  cd HPOlib2
  for i in `cat requirements.txt`; do pip install $i; done
  for i in `cat optional-requirements.txt`; do pip install $i; done
  python setup.py install
  cd ..
  ```

- Notebook requirements:
  ```bash
  conda install -c conda-forge ipympl
  conda install jupyterlab nodejs
  jupyter labextension install @jupyter-widgets/jupyterlab-manager
  ```

- Plotly requirements for jupyterlab:
  
  ```bash
  jupyter labextension install @jupyterlab/plotly-extension
  ```


### Environment

The project uses `.autoenv.zsh` to activate the correct python virtual environment.
By default it uses the name `eth` which can be modified by changing `.autoenv.zsh`.


## Executing Experiment

Please consolidate [`./run.py`](run.py) to see available commands.

Every model configuration is done through a declarative interface to ensure reproductivity and allow for calculating a confidence interval based on the aggregated result of identical model configurations.

If it is required to run the experiment programmatically instead, please see the notebook.
This also illustrates how to recreate the models for already run experiments.


### Executing on Server

The setup is specifically tailored to the [Euler and Leonard cluster at ETH](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) which uses the [IBM LSF batch system](https://scicomp.ethz.ch/wiki/Using_the_batch_system).
To setup the environment follow this [great guide by Tom Stesco](https://www.tomstesco.com/euler-hpc-cluster/).

Whereas a model can be run locally by sending parameters to `run.py`, running it on the server requires running it through a `make` script that ensures proper synchronization.
It automates three steps: 1) pushing code and training data to the server 2) running `run.py` remotely and 3) pulling the results back down so they can be explored in the interactive notebook.

- `make pushdata`: 
  It will push `./raw` and `./processed` to the server which contains training data for the hyperparameter optimization benchmarks. 
  This is required since the servers have blocked access to the network by default. 
  To generate these folder locally first, run one of the models locally by executing `python run.py`.
- `make push`: Used to synchronize code changes.
- `make ARGS="--model gp --n_iter 2" run`: Run this experiment on the server which in the particular case executes BO for two steps using a GP model.
- `make pull`: To pull generated plots and merge the remote CSV database with the local version.

These are all made for Euler. 
For Leonhard use the namespaced version: `make pushdata-leonhard`, `make push-leonhard`, `make run-leonhard` and `make pull-leonhard`.


### Currently run experiments

The currently run experiments are listed in [`tests_euler.txt`](tests_euler.txt) and [`tests_leonhard.txt`](tests_leonhard.txt) for reproductivity and to provide inspiration for possible model configurations. 
The experiments are split in two files so that models benefitting from GPU acceleration can be run on the GPU enabled Leonhard cluster.


## Execute Tests

Run with `make test`. 
Note that currently the end-2-end test requires you to evaluate the plot by sequentially closing them.


## Comparisons

As comparison [Spearmint](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf) and [RoBO](https://bayesopt.github.io/papers/2017/22.pdf).


### RoBO

To run in a linux environment a Docker instance have been created in `./RoBO` with the following `Makefile` commands:

- `make build` Build docker image.
- `make run`   Run container based on image that starts jupyter lab on port `8888`.
          (mounts `/RoBO/shared`)
- `make stop`  Stop container.
- `make start` Start stopped container.
- `make clear` Delete both container and image.


### Spearmint

What follows is a summary of how to use Spearmint (which requires that `mongodb` is installed).
We assume that `./spearmint` is your working directory.

Install (will download to your current directory):

```
git clone https://github.com/HIPS/Spearmint
pip install -e ./Spearmint
```

Run spearmint:

```
mongod --fork --logpath ./log/mongodb.log --dbpath /usr/local/var/mongodb
python ./Spearmint/spearmint/main.py .
```

Quit the mongodb daemon:

1. Find the pid with `top | grep mongo`
2. Kill the process with `kill <pid>`.

To clear mongodb:

```
mongo
use spearmint
db['<experiment_name>.jobs'].remove({status:'pending'})
```

Plot:

```
python spearmint_plots.py .
```

## Resources

Introductory:

- Workshop on GP and BO: https://github.com/gpschool/gprs15b
- Recent tutorial on BO: https://arxiv.org/pdf/1807.02811.pdf

Papers:

- Non-differentiable: https://arxiv.org/pdf/1402.5876.pdf
- DBN learn covariance (unsupervised training): https://papers.nips.cc/paper/3211-using-deep-belief-nets-to-learn-covariance-kernels-for-gaussian-processes.pdf
- NN: https://arxiv.org/pdf/1502.05700.pdf
- KISS-GP: https://arxiv.org/pdf/1511.02222.pdf
- KISS-GP with LOVE: https://arxiv.org/pdf/1803.06058.pdf
- Induced point / Tensor training: https://arxiv.org/pdf/1710.07324.pdf
- Deep Kernel Learning: https://arxiv.org/pdf/1511.02222.pdf
- Ensemble deep kernel learning: https://www-sciencedirect-com.proxy.findit.dtu.dk/science/article/pii/S0169743917307578
- Ensemble kernel learning with NN (not GP!): https://arxiv.org/pdf/1711.05374.pdf
- Mentions embedding in low dimensional subspaces: https://arxiv.org/pdf/1802.07028.pdf
- REMBO (random embedding): https://arxiv.org/pdf/1301.1942.pdf
- Hyperband (comparison with 2xrandom): https://arxiv.org/pdf/1603.06560.pdf
- GP-UCB (Google Vizier implementation): https://arxiv.org/pdf/0912.3995.pdf
- Google Vizier: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf
- Dropout equivalence to GPs: https://arxiv.org/pdf/1506.02142.pdf (1)
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles: https://arxiv.org/pdf/1612.01474v1.pdf (2)
  Implementation: https://github.com/vvanirudh/deep-ensembles-uncertainty
- Batch: http://zi-wang.com/pub/wang-aistats18.pdf
- Marginalize mixture: https://ieeexplore-ieee-org.proxy.findit.dtu.dk/stamp/stamp.jsp?arnumber=5499041
- Horseshoe prior: http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
- lognorm and horseshoe (Snoek): https://arxiv.org/pdf/1406.3896.pdf
- Gamma prior on noise: https://www.researchgate.net/figure/51717248_fig1_Gamma-prior-on-the-total-noise-variance-A-Gamma-prior-is-assumed-for-the-hyperparameter
- HalfT: https://github.com/stan-dev/stan/releases/download/v2.16.0/stan-reference-2.16.0.pdf
- Black Box Optimization Competition: https://bbcomp.ini.rub.de/
