# Training machine learning models

If you are happy with our set of engineered features and want to train different machine learning models to predict stability from them, you can just use the pregenerated csv files we've created and skip down to training ML models.

If you would like to get the initial conditions to generate your own sets of features or make your own predictions (machine learning or analytic), you need to download our dataset of REBOUND simulation\_archives, from which you can extract any dynamical information you like (see <https://rebound.readthedocs.io/>) or run N-body integrations. 

REBOUND integrators are machine independent and we have checked that if you rerun the integrations starting from the simulation\_archives you will get the same answer bit by bit, ensuring you get the exact same instability times despite the dynamics being chaotic. Different sets of integrations were run with different versions of REBOUND, so see `spock/generate_training_data/reproducibility.ipynb` for how to do that.

# Getting the data

First get the data from <https://zenodo.org/record/3723292#.XnvF8i2ZMmU>. Save the tar file to the root spock directory (where this file is). Then

```shell
tar xzvf data.tar.gz
```

will untar the data folder where it needs to be (all scripts assume that this folder is located at `spock/data` so make sure you don't put it somewhere else).

# Generating Features

Different training/testing sets were run at different times with different versions of REBOUND. You first need to clone the REBOUND repository (<https://github.com/hannorein/rebound>) (even if you already installed it with pip).

To regenerate the labels.csv (instability timesfor each example), massratios.csv (planetary masses for each example) and runstrings.csv (names of all the examples) that are needed for subsequent scripts run

```shell
cd /path/to/rebound/
git checkout 6fb912f615ca542b670ab591375191d1ed914672
pip install -e .

cd /path/to/spock/generate_training_data/
python generate_metadata.py
```

For a given function that generates a set of features for a given initial configuration, we have to run a script to generate all those features for all the examples in the training set. 

That script is ``spock/generate_training_data/generate_data.py`` and you specify it by setting runfunc on line 13. Here we will use the ``features`` function actually used by SPOCK as well as the ``additional_features`` function we wrote for all the comparisons to previous work in our paper. We run

```shell
python generate_data.py
```

once for every runfunc for which we want to generate features. This will create and populate a new directory in ``spock/training_data``, with the same name as the runfunc for each of the datasets (will take a while!). Depends on dask for the parallelization.

The above script will run the datasets that were run with the same REBOUND commit given above. After these run, we need to checkout a different REBOUND commit and rerun the same scripts to generate features for the other datasets:

```shell
cd /path/to/rebound/
git checkout 4992313d213b0be717a0b82002e0b89a143c9828
pip install -e .

cd /path/to/spock/generate_training_data
python generate_data.py
```






