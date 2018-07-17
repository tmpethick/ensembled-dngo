FROM ubuntu
RUN apt-get update
RUN apt-get -qq -y install python3-pip build-essential libssl-dev libffi-dev python-dev git tmux
RUN pip3 install --upgrade
RUN pip3 install jupyter jupyterlab

# TODO: Use the path installed by `all_requirements.txt`.
WORKDIR /root/
RUN git clone https://github.com/stokasto/sgmcmc
RUN echo 'export PYTHONPATH=$PYTHONPATH:/root/sgmcmc' >> .bashrc

WORKDIR /root/
RUN apt-get -qq -y install libeigen3-dev swig gfortran
RUN git clone https://github.com/automl/RoBO
WORKDIR /root/RoBO/
# RUN for req in $(cat requirements.txt); do pip3 install $req; done
RUN for req in $(cat all_requirements.txt); do pip3 install $req; done
RUN python3 setup.py install
# TODO: maybe put all_requirements after setup to fix `direct` dependency problem.

EXPOSE 8888
# CMD tmux new 'jupyter lab --ip=0.0.0.0 --no-browser --allow-root /root/shared'
