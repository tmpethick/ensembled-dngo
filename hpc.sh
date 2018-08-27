#!/bin/sh

export PYTHONPATH=$HOME/python/lib64/python3.6/site-packages:$PYTHONPATH
source $HOME/miniconda/bin/activate ""
source $HOME/miniconda/bin/activate $HOME/miniconda/envs/eth ""
cd $HOME/bonn
echo "$@"
python run.py "$@"
