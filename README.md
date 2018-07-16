## Bayesian Optimization

## TODO

- RoBO
- Spearmint 1D (incl. plot)
- RoBO understand weights (pretrained?)
- Implement DNGO


## RoBO

- `build` Build docker image.
- `run`   Run container based on image that starts jupyter lab on port `8888`.
          Use if notebook file is updated on host.
- `stop`  Stop container.
- `start` Start stopped container (used to continue work).
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
