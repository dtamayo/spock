#!/bin/bash
conda create -y --name spock -c pytorch -c conda-forge python=3.8 numpy scipy pandas scikit-learn matplotlib pytorch torchvision torchaudio cudatoolkit=10.1 numba dask xgboost tqdm dill jupyter rebound seaborn fire einops pytorch_lightning &&
    conda activate spock &&
    pip install celluloid icecream celmech 
