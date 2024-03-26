import rebound
import numpy as np
import pandas as pd
import dask.dataframe as dd
import sys
sys.path.append('../spock')
sys.path.append('../../spock')
from simsetup import init_sim_parameters

def get_sim(row, csvfolder):
    try:
        df = pd.read_csv(csvfolder+'/runstrings.csv', header=None, index_col=0, squeeze=True)
        # convert the dataframe into a dictionary
        my_dict = df.to_dict()
        # the binary file name you want to search for
        binary_file_name = row['runstring']
        # search for the binary file name in the dictionary
        for key, value in my_dict.items():
            if value == binary_file_name:
                i = int(key)
        ics = np.loadtxt(csvfolder+'/initial_conditions.csv', delimiter=',',dtype=np.float64)
        # get corresponding row
        data = ics[i]
        # create a new simulation
        sim = rebound.Simulation()
        sim.G=4*np.pi**2
        sim.add(m=data[0], x=data[1], y=data[2], z=data[3], vx=data[4], vy=data[5], vz=data[6])
        sim.add(m=data[7], x=data[8], y=data[9], z=data[10], vx=data[11], vy=data[12], vz=data[13])
        sim.add(m=data[14], x=data[15], y=data[16], z=data[17], vx=data[18], vy=data[19], vz=data[20])
        sim.add(m=data[21], x=data[22], y=data[23], z=data[24], vx=data[25], vy=data[26], vz=data[27])
        return sim
 
    except:
        print("training_data_functions.py Error reading initial condition {0}".format(i))
        return None

def training_data(row, csvfolder, runfunc, args):
    try:
        sim = get_sim(row, csvfolder)
        init_sim_parameters(sim)
        ret, stable = runfunc(sim, args)
    except:
        print('{0} failed'.format(row['runstring']))
        return None

    r = ret[0] # all runfuncs return list of features for all adjacent trios (to not rerun for each). For training assume it's always 3 planets so list of 1 trio
    return pd.Series(r, index=list(r.keys())) # convert OrderedDict to pandas Series

def gen_training_data(outputfolder, csvfolder, runfunc, args):
    # assumes runfunc returns a pandas Series of features, and whether it was stable in short integration. See features fucntion in spock/feature_functions.py for example
    df = pd.read_csv(csvfolder+"/runstrings.csv", index_col = 0).head(100)
    ddf = dd.from_pandas(df, npartitions=48)
    testres = training_data(df.loc[0], csvfolder, runfunc, args) # Choose formatting based on selected runfunc return type
    
    metadf = pd.DataFrame([testres]) # make single row dataframe to autodetect meta
    res = ddf.apply(training_data, axis=1, meta=metadf, args=(csvfolder, runfunc, args)).compute(scheduler='processes')
    res.to_csv(outputfolder+'/trainingdata.csv')

