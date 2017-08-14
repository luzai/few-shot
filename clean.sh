#!/usr/bin/env bash
rm output* -rf
find . -name '*.pyc' -exec rm {} \;
find . -name '*.pdf' -exec rm {} \;
find . -name '*.png' -exec rm {} \;
find . -name '*.log' -exec rm {} \;
find . -name 'dbg' -exec rm {} \;
rm tmp.pkl tmp.png *.log *.pkl *.pyc -f
#cat ~/.bash_history | head -n 130 > ~/.bash_history


#conda install pip setuptools graphviz pydot nomkl numpy scipy scikit-learn numexpr moviepy
##conda remove mkl mkl-service
#conda install  -c conda-forge pathos
#pip install -U keras  tensorflow-gpu tensorflow-tensorboard gputil