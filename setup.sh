#!/usr/bin/bash

################################################################################
# old version
################################################################################
# # virtual env
# conda create -n ogw python=3.7 -y
# conda activate ogw

# # install packages
# conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
# pip install torch-geometric pot

# # install subplementary
# conda install matplotlib seaborn joblib networkx numba ipykernel -y
# # pip install qpsolvers nsopy

# python setup.py develop

################################################################################
# new version
################################################################################

conda create -n ogw python=3.8 -y
conda activate ogw
conda install pytorch torchvision torchaudio cudatoolkit=11.3 \
              pyg autopep8 flake8 ipython matplotlib seaborn joblib networkx numba ipykernel ipywidgets \
              -c pytorch \
              -c pyg \
              -y
pip install pot
python setup.py develop
