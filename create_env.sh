#!/bin/bash
conda create -y --name spock -c pytorch -c conda-forge python=3.7 numpy scipy pandas scikit-learn matplotlib pytorch=1.5.1 torchvision torchaudio cudatoolkit=10.1 numba dask xgboost=1.2.0 tqdm dill jupyter rebound seaborn fire einops h5py && ~/miniconda3/envs/spock/bin/pip install celluloid icecream celmech pytorch_lightning
