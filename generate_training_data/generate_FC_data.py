# %% [markdown]
# # Generate SPOCK training data

# %%
import spock
import random
import numpy as np
import rebound
import pandas as pd
from spock import simsetup
from spock import FeatureClassifier

# %% [markdown]
# The initial conditions are stored as snapshots of a simulation archive, we must thus load the datapath and the labels for the corresponding systems

# %%
#specify the data path
#We will be using cleaned data generated from the original spock initial conditions data
# This data is in the form of a simulation archive
datapath = '../../cleanData/csvs/resonant/'
labels = pd.read_csv(datapath+'sim_labels.csv')

# %% [markdown]
# We can now generate the set of system indices based on the labels

# %%
#generates the indexes of the systems
systemNum = range(labels.shape[0])

# %% [markdown]
# We can note the column names and import the different feature generators

# %%
spock = FeatureClassifier()

# %% [markdown]
# We can then establish some helper functions that will allow us to map the spock.generate_feature function to the different systems by mapping to the different snapshots

# %%
def getList(features):
    '''Helper function which isolates the data list from the generate_features return'''
    return list(features[0][0].values())+[features[1]]

# %%
def getFeat(num):
    '''when given a index of a row, loads initial conditions and returns the spock generated features'''
    #gets features based on index num
    sim = rebound.Simulation(datapath+"clean_initial_conditions.bin", snapshot=num)
    return spock.generate_features(sim)
# %%
rebound.__version__

# %% [markdown]
# We can now map getFeat to the different rows of the Initial df, this will create each simulation and generate the spock features.

# %%
import sys
from multiprocessing import Pool
if __name__ == "__main__":
    with Pool() as pool:
        features = pool.map(getFeat,systemNum)
        pool.close()
        pool.join()
#formats the data correctly
#formats the data correctly regardless of features
#%%
formattedFeat = pd.DataFrame(np.array(list(map(getList,features))), 
                             columns = list(features[0][0][0].keys())+['InitialStable'])


# %% [markdown]
# We can then join the generated features with the corresponding labels

# %%
dataset = pd.DataFrame.join(formattedFeat,labels)

# %% [markdown]
# We can then save the new training data spreadsheet.

# %%
dataset.to_csv(datapath+'2pTestData.csv')


