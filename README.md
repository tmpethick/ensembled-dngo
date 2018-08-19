## Bayesian Optimization

Good resource: https://github.com/gpschool/gprs15b

## The Article Archives

- Recent tutorial on BO (July 2018): https://arxiv.org/pdf/1807.02811.pdf
- Non-stationary: [3], [22], [54], [70], [118], [121], [126], [132], [141]
- Non-differentiable: https://arxiv.org/pdf/1402.5876.pdf
- DBN learn covariance (unsupervised training): https://papers.nips.cc/paper/3211-using-deep-belief-nets-to-learn-covariance-kernels-for-gaussian-processes.pdf
- NN: https://arxiv.org/pdf/1502.05700.pdf
- KISS-GP: https://arxiv.org/pdf/1511.02222.pdf !!!!
- KISS-GP with LOVE: https://arxiv.org/pdf/1803.06058.pdf
- Induced point / Tensor training: https://arxiv.org/pdf/1710.07324.pdf
- Deep Kernel Learning (Add RBF/Spectral mixture covariance instead of linear): https://arxiv.org/pdf/1511.02222.pdf
- Ensemble deep kernel learning: https://www-sciencedirect-com.proxy.findit.dtu.dk/science/article/pii/S0169743917307578
- Ensemble kernel learning with NN (not GP!): https://arxiv.org/pdf/1711.05374.pdf
- Mentions embedding in low dimensional supspaces: https://arxiv.org/pdf/1802.07028.pdf
- REMBO (embedding): https://arxiv.org/pdf/1301.1942.pdf

- Dropout equivalence to GPs: https://arxiv.org/pdf/1506.02142.pdf (1)
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles: https://arxiv.org/pdf/1612.01474v1.pdf (2)
  Implementation: https://github.com/vvanirudh/deep-ensembles-uncertainty

- Batch: http://zi-wang.com/pub/wang-aistats18.pdf
- Marginalize mixture: https://ieeexplore-ieee-org.proxy.findit.dtu.dk/stamp/stamp.jsp?arnumber=5499041
- Horseshoe prior: http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
- lognorm and horseshoe (Snoek): https://arxiv.org/pdf/1406.3896.pdf


Problems:

Problem 1: don't use uncertainty estimate.. (mean is almost identical to acq)
Problem 2: Unprecise about local behaviour. When exploiting it's still done in a relatively random fashion locally.
Problem 3: Not exploring enough.. (only locally)
Problem 4: Uncertainty too big (explores too much) (random flucturations in nn fitting)


Hyperparameters:
- MAP / Marginalize hyperparameters
- Batch size 
  - big is smooth. What for noisy functions?
  - How it works for big n when bath size become relatively small.
- How big an ensemble?

TODO:
- Setup EULER √
- Run all HPOLib √ (2018-08-14 15:35:09.554127)
- Explore results (groupby obj_func, load observations and plot regret) √
- uuid selector √
- Configurable Wall time √
- Run multiple √ (all 3, mcmc 2)
- Hyperparameter optimization of Logistic regression for MNIST
  - Basic run √ (with 10 epochs)
  - make it compatible with FEBO
  - Round after every BFGS
- Confidence interval on regret √
- Aggregate mean and variance √
- setup Bohamiann
- Run on embedded
- Include in FEBO (3 days)
- Write (5 days)
- plot 1D and 2D embeddings
- (Make BFGS depend on dimension)

"We recall a key characteristic of the acquisition
function that they are mostly flat functions with only a few
peaks." – http://proceedings.mlr.press/v80/lyu18a/lyu18a.pdf
(problem in high dim where optimization of acq from random init can't escape large flat region)

Remember to give all models the same computational budget.

- Find cases where RBF have problems.. (non-stationarity, jaggy/discrete)
- metric transformation (euclidean does not work in high dimension)

- Hyperparameter optimization on camelback and goldsteinprice (since both has high different from GP - room for improvement. And goldsteinprice especially could do better overall)
  - variance increases as epochs increases. 100, 1000, 10000. (Testing on different dimensions where exploitation is important: levy, sintwo, hartmann3 in which low variance is visible through low exploration (i.e. high exploitation).)
  - Test weight regularization on goldsteinprice. (0.01, 0.001, 0.0001). Very little suggested by DNGO.
  - Embedding sinone 1D, embedded branin 3D, embedded branin with linear weight (also see embedded rosenbrock10).
  - High dim Rosenbrock10/Ackley10
  - [WAIT] median, median vs maximum
  - [WAIT] Consider that mcmc is maybe necessary..

bohachevsky                    0,-1
branin                         -2,-6
camelback                      -3,-6
forrester                      -6,-8
goldsteinprice                 1,-2
hartmann3                      -3,-5
hartmann6                      -1,-2
levy                           -6
logistic_regression_mnist      -1
rosenbrock                     0,-2
sinone                         -2,-10
sintwo                         -2,-5

- why dngo works
  - Same random initialisation for comparison √
  - In embedding (where nn shines) (plot 1D and 2D embedding)
    - Compare with REMBO (what about rotation?)
  - Random exploration (high peaks sometimes by nn) (see levy 5nn)
  - Epochs effect 
  - Weight regularization (l2)
- ensemble
  - Median vs maximum
  - amount of ensembles
  - Different nn? (epochs, l2)
  - analysis about number of steps in best case?
- Noise? (if noise is not fixed then non-differentiable problems will just expand the overall uncertainty.)
- high dim experiment
  - Rosenbrock10? Ackley10? (see Picheny et al. for Rosenbrock10)
    below 10D: random sample 20,  iter 45
    above 10D: random sample 100, iter 175.
    http://proceedings.mlr.press/v80/lyu18a/lyu18a.pdf
  - No more than 10 since otherwise we would run into problems with optimization the acquisition function.
  - Minibatches (necessary for many datapoint. Explore how they effect in small sample size. adjust learning rate accordingly)
  - For many datapoints in high dimensions (mnist)



## Results

bohachevsky     same
branin          median better, max slightly better
levy            dngo converges quicker (other no difference)
sintwo          median better, max worse
sinone          dngo stuck, median stuck up until end, max better than gp
camelback       same
rosenbrock      dngo best by small margin, max worse, median second best
forrester       dngo stuck (random init fault?), max and median equally good and better than gp
goldsteinprice  same
hartmann3       max and median converges faster (max best), not complete
hartmann6       same, not complete

## Linear in O(n)

"The equations for linear regression can be performed in primal form (e.g. the regular normal equations), where there is one parameter per input variable, or dual form (e.g. kernel ridge regression with a linear kernel) where there is one parameter per training example. This means you can choose which form to use depending on which is more efficient, if N>>d then use the normal equation, which is O(N), if d>>N then use kernel ridge regression with a linear kernel, where d is the number of attributes. Bayesian linear regression is to GP with a linear covariance function what linear regression is to kernel ridge regression with a linear kernel."

## RoBO

- `build` Build docker image.
- `run`   Run container based on image that starts jupyter lab on port `8888`.
          (mounts `/RoBO/shared`)
- `stop`  Stop container.
- `start` Start stopped container.
- `clear` Delete both container and image.


## Spearmint

Environment (everything happens inside `/spearmint`):

```
cd spearmint
```

Install (will change your current dir):

```
cd ../.. && git clone https://github.com/HIPS/Spearmint
pip install -e ./Spearmint
```

Run spearmint:

```
mongod --fork --logpath ./log/mongodb.log --dbpath /usr/local/var/mongodb
source activate spearmint
python ../../Spearmint/spearmint/main.py .
```

Quit the daemon:

* Find the pid with `top | grep mongo`
* Kill the process with `kill <pid>`.

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


## Env


Note: incomplete

```
conda create -n eth python=3.6
source $HOME/miniconda/bin/activate
source activate eth
conda install -y pytorch-cpu torchvision-cpu -c pytorch
conda install -y -c conda-forge blas seaborn scipy matplotlib pandas gpy pathos emcee
git clone https://github.com/automl/HPOlib2.git
cd HPOlib2
for i in `cat requirements.txt`; do pip install $i; done
for i in `cat optional-requirements.txt`; do pip install $i; done
python setup.py install
cd ..
```

Notebook requirements:

```
conda install -c conda-forge ipympl
conda install jupyterlab nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Plotly requirement in jupyterlab:

```
jupyter labextension install @jupyterlab/plotly-extension
```

`.autoenv.zsh`:

```
source activate <name>
```


## Euler

Euler setup see: https://www.tomstesco.com/euler-hpc-cluster/

e.g.

```
make ARGS="--model gp --n_iter 2" run
```

To pull files and merge csv database:

```
make pull
```

## Currently run experiments

```
make ARGS="--model gp --n_iter 200 -f branin" run
make ARGS="--model gp --n_iter 200 -f hartmann3" run
make ARGS="--model gp --n_iter 200 -f hartmann6" run
make ARGS="--model gp --n_iter 200 -f camelback" run
make ARGS="--model gp --n_iter 200 -f forrester" run
make ARGS="--model gp --n_iter 200 -f bohachevsky" run
make ARGS="--model gp --n_iter 200 -f goldsteinprice" run
make ARGS="--model gp --n_iter 200 -f levy" run
make ARGS="--model gp --n_iter 200 -f rosenbrock" run
make ARGS="--model gp --n_iter 200 -f sinone" run
make ARGS="--model gp --n_iter 200 -f sintwo" run

make ARGS="--model dngo --n_iter 200 -f branin" run
make ARGS="--model dngo --n_iter 200 -f hartmann3" run
make ARGS="--model dngo --n_iter 200 -f hartmann6" run
make ARGS="--model dngo --n_iter 200 -f camelback" run
make ARGS="--model dngo --n_iter 200 -f forrester" run
make ARGS="--model dngo --n_iter 200 -f bohachevsky" run
make ARGS="--model dngo --n_iter 200 -f goldsteinprice" run
make ARGS="--model dngo --n_iter 200 -f levy" run
make ARGS="--model dngo --n_iter 200 -f rosenbrock" run
make ARGS="--model dngo --n_iter 200 -f sinone" run
make ARGS="--model dngo --n_iter 200 -f sintwo" run

make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f branin" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f hartmann3" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f hartmann6" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f camelback" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f forrester" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f bohachevsky" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f goldsteinprice" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f levy" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f rosenbrock" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f sinone" run
make W="20:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f sintwo" run

make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f branin" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f hartmann3" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f hartmann6" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f camelback" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f forrester" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f bohachevsky" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f goldsteinprice" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f levy" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f rosenbrock" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f sinone" run
make W="20:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f sintwo" run

make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f branin" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f hartmann3" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f hartmann6" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f camelback" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f forrester" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f bohachevsky" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f goldsteinprice" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f levy" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f rosenbrock" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f sinone" run
make W="20:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f sintwo" run

make W="24:00" ARGS="--model gp --n_iter 200 -f logistic_regression_mnist" run
make W="24:00" ARGS="--model dngo --n_iter 200 -f logistic_regression_mnist" run
make W="24:00" ARGS="--model dngo -nn 5 -agg median --n_iter 200 -f logistic_regression_mnist" run
make W="24:00" ARGS="--model dngo -nn 5 -agg max --n_iter 200 -f logistic_regression_mnist" run
make W="24:00" ARGS="--model dngo -mcmc 20 --n_iter 200 -f logistic_regression_mnist" run
```

