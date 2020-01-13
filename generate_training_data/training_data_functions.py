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
        ret = runfunc(sim, args)
    except:
        print('{0} failed'.format(row['runstring']))
        return None
    try:
        return ret[0]
    except:
        return ret

def gen_training_data(outputfolder, safolder, runfunc, args):
    df = pd.read_csv(outputfolder+"/runstrings.csv", index_col = 0)
    ddf = dd.from_pandas(df, npartitions=24)
    sa = rebound.SimulationArchive(safolder+'sa'+df.loc[0]['runstring'])
    testres = runfunc(sa[0], args) # Choose formatting based on selected runfunc return type
    
    if isinstance(testres, np.ndarray): # for runfuncs that return an np array of time series
        res = ddf.apply(training_data, axis=1, meta=('f0', 'object'), args=(safolder, runfunc, args)).compute(scheduler='processes') # dask meta autodetect fails. Here we're returning a np.array not Series or DataFrame so meta = object
        Nsys = df.shape[0]
        Ntimes = res[0].shape[0]
        Nvals = res[0].shape[1] # Number of values at each time (18 for orbtseries, 6 per planet + 1 for time)
        matrix = np.concatenate(res.values).ravel().reshape((Nsys, Ntimes, Nvals)) 
        np.save(outputfolder+'/trainingdata.npy', matrix)

    if isinstance(testres, pd.Series):
        metadf = pd.DataFrame([testres]) # make single row dataframe to autodetect meta
        res = ddf.apply(training_data, axis=1, meta=metadf, args=(safolder, runfunc, args)).compute(scheduler='processes')
        # meta autodetect should work for simple functions that return a series
        res.to_csv(outputfolder+'/trainingdata.csv')

    if isinstance(testres, list):
        metadf = pd.DataFrame([testres[0]]) # make single row dataframe to autodetect meta
        res = ddf.apply(training_data, axis=1, meta=metadf, args=(safolder, runfunc, args)).compute(scheduler='processes')
        # meta autodetect should work for simple functions that return a series
        res.to_csv(outputfolder+'/trainingdata.csv')
