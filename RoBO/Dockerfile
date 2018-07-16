FROM ubuntu
RUN apt-get update
RUN apt-get -qq -y install python-pip python-dev build-essential git
RUN pip install --upgrade pip==9.0.3
RUN pip install jupyter jupyterlab

# RUN apt-get -qq -y install libeigen3-dev swig gfortran
# RUN git clone https://github.com/automl/RoBO
# WORKDIR RoBO/
# RUN for req in $(cat requirements.txt); do pip install $req; done
# RUN for req in $(cat all_requirements.txt); do pip install $req; done
# RUN python setup.py install

EXPOSE 8888
WORKDIR $HOME/
CMD jupyter lab --ip=0.0.0.0 --no-browser --allow-root
