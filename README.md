## Bayesian Optimization

Good resource: https://github.com/gpschool/gprs15b

## The Article Archives

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

Tests to run:
- Prior (logGaussian(0, 1) for alpha), fixed zero noise)
- Domains: discrete domain, Non-stationarity

Benchmarks:

             DNGO | GP      | Note
Hartmann6 | 10^-2 | 10^-1.5 | DNGO outperform GP
sinOne    | 10^-6 | 10^-8   | 
Hartmann3 | 10^-4 | 10^-6   | 
Branin    | 10^-3 | 10^-5   | 

Theory Questions:
- How does data outweight prior with more data
- GP assumption


TODO:
- Fix folder name
- Fix gpy mcmc
- Improve DNGO to compare with GP
- Remove batch completely? Seems to make it instable.

- Removing data: doesn't this lead to miscalibrated uncertainty estimates? => prediction wrong about something of which it should be certain. (inspired by (2))
- Test Dropout? (Inspired by (1) and (2))
- Need to train for longer to overfit/be different. Is this good/desireable?
- Better for e.g. noisy data?
- Why is the speed dropping dramatically?

- Find case in which it does not work (modified branin with drop in unexplored area)
- Ensemble using max, average, median
- Felix benchmark functions


- Loop training networks (basis functions) (consider paralizability)
- GP on each
- BOModel specifies in `acq` how to aggregate 
  (calc for all, then call `ensemble_aggregator`)
- 

Hyperparameters:
- MAP / Marginalize hyperparameters
- Batch size 
  - big is smooth. What for noisy functions?
  - How it works for big n when bath size become relatively small.
- How big an ensemble?

Testing:
- Test wider input domain (np.sinc(x * 10 - 5).sum(axis=1)[:, None] * 100000 on [0,10] interval) (i.e. spike)
- noise
- Play with relu and l2

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
conda create -n <name> python=3.6 scipy jupyterlab matplotlib
conda install -c conda-forge ipympl
conda install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager

git clone https://github.com/automl/HPOlib2.git
cd HPOlib2
for i in `cat requirements.txt`; do pip install $i; done
for i in `cat optional_requirements.txt`; do pip install $i; done
python setup.py install
cd ..
```

Plotly requirement in jupyterlab:

```
jupyter labextension install @jupyterlab/plotly-extension
```

`.autoenv.zsh`:

```
source activate <name>
```
