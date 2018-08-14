#!/bin/sh

module load new gcc/4.8.2 python/3.6.1
export PYTHONPATH=$HOME/python/lib64/python3.6/site-packages:$PYTHONPATH
source $HOME/miniconda/bin/activate ""
source $HOME/miniconda/bin/activate $HOME/miniconda/envs/eth ""
cd $HOME/bonn
echo "$@"
python run.py "$@"
