import rebound
import numpy as np
import pandas as pd
import dask.dataframe as dd

def training_data(row, safolder, runfunc, args):
    try:
        sa = rebound.SimulationArchive(safolder+'sa'+row['runstring'])
        sim = sa[0]
    except:
        print("traininst_data_functions.py Error reading " + safolder+'sa'+row['runstring'])
        return None

    try:
        ret, stable = runfunc(sim, args)
    except:
        print('{0} failed'.format(row['runstring']))
        return None

    return ret[0] # all runfuncs return list of features for all adjacent trios (to not rerun for each). For training assume it's always 3 planets so list of 1 trio

def gen_training_data(outputfolder, safolder, runfunc, args):
    # assumes runfunc returns a pandas Series of features, and whether it was stable in short integration. See features fucntion in spock/feature_functions.py for example
    df = pd.read_csv(outputfolder+"/runstrings.csv", index_col = 0)
    ddf = dd.from_pandas(df, npartitions=24)
    testres = training_data(df.loc[0], safolder, runfunc, args) # Choose formatting based on selected runfunc return type

    metadf = pd.DataFrame([testres]) # make single row dataframe to autodetect meta
    res = ddf.apply(training_data, axis=1, meta=metadf, args=(safolder, runfunc, args)).compute(scheduler='processes')
    res.to_csv(outputfolder+'/trainingdata.csv')
