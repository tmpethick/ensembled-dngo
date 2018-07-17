## Bayesian Optimization

- Non-stationary: [3], [22], [54], [70], [118], [121], [126], [132], [141]
- Non-differentiable: https://arxiv.org/pdf/1402.5876.pdf
- DBN learn covariance (unsupervised training): https://papers.nips.cc/paper/3211-using-deep-belief-nets-to-learn-covariance-kernels-for-gaussian-processes.pdf
- NN: https://arxiv.org/pdf/1502.05700.pdf
- KISS-GP: https://arxiv.org/pdf/1511.02222.pdf !!!!
- KISS-GP with LOVE: https://arxiv.org/pdf/1803.06058.pdf
- Induced point / Tensor training: https://arxiv.org/pdf/1710.07324.pdf
- Deep Kernel Learning (Add RBF/Spectral mixture covariance instead of linear): https://arxiv.org/pdf/1511.02222.pdf

## Meeting 

- Done so far
  - Experimented with RoBO and Spearmint
    - (*) run both on simple example that fails
  - ML (point estimate)
  - Marginalize hyperparams using HMC (with gamma prior)
  - DNGO network
  - DNGO in BO
- Found interesting approaches by Andrew Gordon Wilson (DKL) and Induced point
  - (*) Print and bring
  - Does not deal with BO however (few datapoints)
- Goal of project?
  - BO for many datapoints?
  - Non-stationarity?
- How to do ensemble?
- Problem with getting small network to work
- Look into sparse GP?
- Access to cluster 
  - Right now I have setup docker instance
  - Do not need it now but good to know how it is structured.

## Questions

- RoBO understand weights (pretrained?)


## TODO

- RoBO and Spearmint on 1d example (find problems)
- Implement basis function regression
- make sure basis function leads to positive semi-definite matrix..
- Implement DNGO


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
```

`.autoenv.zsh`:

```
source activate <name>
```
