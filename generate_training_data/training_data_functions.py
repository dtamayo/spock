import rebound
import numpy as np
import pandas as pd
import dask.dataframe as dd
from spock.simsetup import set_sim_parameters, rescale, set_timestep

def training_data(row, safolder, runfunc, args):
    try:
        sa = rebound.SimulationArchive(safolder+'sa'+row['runstring'])
        sim = sa[0]
    except:
        print("traininst_data_functions.py Error reading " + safolder+'sa'+row['runstring'])
        return None

    sim = rescale(sim)
    set_sim_parameters(sim)
    set_timestep(sim, dtfrac=0.06)

    try:
        ret, stable = runfunc(sim, args)
    except:
        print('{0} failed'.format(row['runstring']))
        return None

    r = ret[0] # all runfuncs return list of features for all adjacent trios (to not rerun for each). For training assume it's always 3 planets so list of 1 trio
    return pd.Series(r, index=list(r.keys())) # conert OrderedDict to pandas Series

def gen_training_data(outputfolder, safolder, runfunc, args):
    # assumes runfunc returns a pandas Series of features, and whether it was stable in short integration. See features fucntion in spock/feature_functions.py for example
    df = pd.read_csv(outputfolder+"/runstrings.csv", index_col = 0)
    ddf = dd.from_pandas(df, npartitions=48)
    testres = training_data(df.loc[0], safolder, runfunc, args) # Choose formatting based on selected runfunc return type

    metadf = pd.DataFrame([testres]) # make single row dataframe to autodetect meta
    res = ddf.apply(training_data, axis=1, meta=metadf, args=(safolder, runfunc, args)).compute(scheduler='processes')
    res.to_csv(outputfolder+'/trainingdata.csv')
