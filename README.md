## Bayesian Optimization

## The Article Archives

- Non-stationary: [3], [22], [54], [70], [118], [121], [126], [132], [141]
- Non-differentiable: https://arxiv.org/pdf/1402.5876.pdf
- DBN learn covariance (unsupervised training): https://papers.nips.cc/paper/3211-using-deep-belief-nets-to-learn-covariance-kernels-for-gaussian-processes.pdf
- NN: https://arxiv.org/pdf/1502.05700.pdf
- KISS-GP: https://arxiv.org/pdf/1511.02222.pdf !!!!
- KISS-GP with LOVE: https://arxiv.org/pdf/1803.06058.pdf
- Induced point / Tensor training: https://arxiv.org/pdf/1710.07324.pdf
- Deep Kernel Learning (Add RBF/Spectral mixture covariance instead of linear): https://arxiv.org/pdf/1511.02222.pdf

- Batch: http://zi-wang.com/pub/wang-aistats18.pdf
- Marginalize mixture: https://ieeexplore-ieee-org.proxy.findit.dtu.dk/stamp/stamp.jsp?arnumber=5499041

TODO:
- save model
- predict batch √
- Pre-train (on what?) and no training during BO.
- Reproduce results from RoBO
- linear regression (no need for cholesky decomp.)


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
```

`.autoenv.zsh`:

```
source activate <name>
```
