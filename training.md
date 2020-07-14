# Training machine learning models or running comparisons

In order to retrain our machine learning models, to train your own, or to test any other type of model against our data, you need to download our dataset of REBOUND SimulationArchives, from which you can extract the initial conditions and any dynamical information you like (see <https://rebound.readthedocs.io/>) or run N-body integrations.

REBOUND integrators are machine independent and we have checked that if you rerun the integrations starting from the SimulationArchives you will get the same answer bit by bit, ensuring you get the exact same instability times despite the dynamics being chaotic. Different sets of integrations were run with different versions of REBOUND, so see `spock/generate_training_data/reproducibility.ipynb` for how to do that.

# Getting the data

First get the data from <https://zenodo.org/record/3723292#.XnvF8i2ZMmU>. Save the tar file to the root spock directory (where this file is). Then

```shell
tar xzvf data.tar.gz
```

will untar the data folder where it needs to be (all scripts assume that this folder is located at `spock/data` so make sure you don't put it somewhere else).

# Required REBOUND versions

Different training/testing sets were run at different times with different versions of REBOUND. You first need to clone MY FORK of the REBOUND repository (wherever you like) 

```shell
cd /path/to/wherever/you/want
git clone https://github.com/dtamayo/rebound.git
```

# Reading the instability times for the integrations in each dataset

To regenerate the labels.csv (instability times for each example), massratios.csv (planetary masses for each example) and runstrings.csv (names of all the examples) that are needed for subsequent scripts run

```shell
cd /path/to/rebound/
git checkout 6fb912f615ca542b670ab591375191d1ed914672
pip install -e .

cd /path/to/spock/generate_training_data/
python generate_metadata.py
```

The instability times generated in `training_data/<nameofdataset>/labels.csv, together with the corresponding simulation archives in data/ should be sufficient to evaluate the performance of any new model on our datasets.

# Retraining or modifying SPOCK

In order to train the model, you first have to regenerate the features we use for SPOCK for each initial condition in our test sets.
For a given function that generates a set of features for a given initial configuration, we have to run a script to generate all those features for all the examples in the training set. 

That script is `spock/generate_training_data/generate_data.py` and you specify the corresponding function (defined in `spock/spock/feature_funcctions.py` by setting runfunc on line 13. You can just run the `features` function actually used by SPOCK, but if you want to regenerate our figures and do comparisons you also have to run the `additional_features` function. After editing line 13 to what you want,

```shell
python generate_data.py
```

once for every runfunc for which we want to generate features. This will create and populate a new directory in ``spock/training_data``, with the same name as the runfunc for each of the datasets (will take a while!). Depends on dask for the parallelization.

The above script will run the datasets that were run with the same REBOUND commit given above. After these run, we need to checkout a different REBOUND commit and rerun the same scripts to generate features for the other datasets:

```shell
cd /path/to/rebound/
git checkout 4a6c79ae14ffde27828dd9d1f8d8afeba94ef048 
pip install -e .

cd /path/to/spock/generate_training_data
python generate_data.py
```

# Retraining SPOCK

Once you have generated all the features, you can retrain SPOCK by running the `train_models/train_spock.ipynb` notebook. To train the comparison models used in the paper, run `train_models/train_comparison_models.ipynb`. This will save the trained models in `spock/spock/models`.

# To regenerate the figures

First

```shell
cd /path/to/rebound/
git checkout 6fb912f615ca542b670ab591375191d1ed914672
pip install -e .

cd /path/to/spock/paper_plots
python generatePratios.py
```
After that you should be able to run all the figure notebooks (note Figs 6,7 take about 20 mins, Fig 5 would take about 20 times longer on a single core but runs over all available cores).
