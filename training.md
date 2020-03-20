# Getting the data

First get the data from zenodo and put the 'data' directory in the root spock directory (where this file is, spock/data)

# Generating Features

Different training/testing sets were run at different times with different versions of REBOUND. You need to clone the REBOUND repository (<https://github.com/hannorein/rebound>) then

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






