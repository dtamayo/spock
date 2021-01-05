conda create -y --name spock -c pytorch -c conda-forge python=3.8 numpy scipy pandas scikit-learn matplotlib pytorch torchvision torchaudio cudatoolkit=10.1 numba dask xgboost tqdm dill jupyter
conda activate spock
pip install celluloid icecream einops fire celmech rebound==3.9.0 pytorch_lightning==0.9.0 seaborn==0.11.0
