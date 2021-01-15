#!/bin/bash
conda create -q -y --name $1 -c pytorch -c conda-forge python=3.7 numpy scipy pandas scikit-learn matplotlib torchvision pytorch xgboost rebound>=3.14.0 einops jupyter pytorch-lightning ipython h5py
