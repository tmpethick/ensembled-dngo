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

- Dropout equivalence to GPs: https://arxiv.org/pdf/1506.02142.pdf (1)
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles: https://arxiv.org/pdf/1612.01474v1.pdf (2)
  Implementation: https://github.com/vvanirudh/deep-ensembles-uncertainty

- Batch: http://zi-wang.com/pub/wang-aistats18.pdf
- Marginalize mixture: https://ieeexplore-ieee-org.proxy.findit.dtu.dk/stamp/stamp.jsp?arnumber=5499041
- Horseshoe prior: http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
-lognorm and horseshoe (Snoek): https://arxiv.org/pdf/1406.3896.pdf


How it was done:
- Finding good weight decay:
  - use gpy points
  - run one step of DNGO BO
  - Plot the mean and acq restricted to the exploited area

Problems:

Problem 1: don't use uncertainty estimate.. (mean is almost identical to acq)
Problem 2: Unprecise about local behaviour. When exploiting it's still done in a relatively random fashion locally.
Problem 3: Not exploring enough.. (only locally)
Problem 4: Uncertainty too big (explores too much)

- Should be more certain in areas with many points.
- mcmc * ensemble: should we some how average the ensemble before mcmc to get mcmc + ensemble?
- test performance on massive sample?

TODO:
- Removing data: doesn't this lead to miscalibrated uncertainty estimates? => prediction wrong about something of which it should be certain. (inspired by (2))
- Test Dropout? (Inspired by (1) and (2))
- Need to train for longer to overfit/be different. Is this good/desireable?
- Better for e.g. noisy data?
- Why is the speed dropping dramatically?

Benchmarks:

             DNGO | GP      | Note
Hartmann6 | 10^-2 | 10^-1.5 | DNGO outperform GP
sinOne    | 10^-6 | 10^-8   | 
Hartmann3 | 10^-4 | 10^-6   | 
Branin    | 10^-3 | 10^-7   | 

Hyperparameters:
- MAP / Marginalize hyperparameters
- Batch size 
  - big is smooth. What for noisy functions?
  - How it works for big n when bath size become relatively small.
- How big an ensemble?

Okay, a small recap of my embedded experiments. I tried it on an embedded sinc and GP clearly have problems (see attachments). However, with Automatic Relevance Detection it obviously quickly discard the irrelevant dimension.
NN is almost identical to GP with ARD after 20 iterations (see attachment).

My concern is that even though this experiment shows the usefulness of NN *it does not provide a failure case for NN that would justify an ensemble*.

But as the attachment shows it did seem like the NN could get stuck if there were multiple local optima. It was indeed the case for embedded sinOne. An ensemble might help it escape more quickly. However, don't we at most save a factor k steps where k is the number of NN in the ensemble? Assuming the BO is stuck (thus resampling approximately the incumbent point) the neural networks over the next k steps are trained on almost the same data as an ensemble of k networks in one step.
Each step takes linear time to train so k*n steps takes O((k*n)^2) time. Using ensemble of k this becomes O(k * n^2) so we save a factor k. Is this really big enough to bother?

I'm considering trying other methods to improve the uncertainty estimates. One alternative is training the nn and gp jointly like Wilson (Deep Kernel Learning). I haven't seen it done for BO, but this work might overlap too much with their research grant/work?

Best
Thomas

PS: I'll do some more thorough experimentation now that Euler is back up (ensemble size, ensemble aggregator, embedding size) depending on your response.. 

- NN could maybe get stuck in local maximum. Is it more probable than in 1D though?
- propose:
  - Rotate embedded f? (f(x + x')) (GP is bad at rotation)
  - still deals with GP weaknesses

- Problems:
  - DNGO exploring corners

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
- Confidence interval on regret
- Aggregate mean and variance
- setup Bohamiann
- Run on embedded
- Include in FEBO (3 days)
- Write (5 days)

Extra:
- plot 1D and 2D embeddings
- (Make BFGS depend on dimension)
- Construct examples
  - Many local max
  - Hidden max
- RL for hyperparameter opt how?

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
