import dill
import pandas as pd
import os
import numpy as np
from  . import featurefunctions
import os

class spockClassifier():
    def __init__(self, modelname='spocktrio_resonant.pkl'):
        spockpath = os.path.dirname(__file__)
        self.model, self.features, featurefuncname = dill.load(open(spockpath+'/models/'+modelname, "rb"))
        self.featurefunc = getattr(featurefunctions, 'spock_features')
    def predict(self, sim, indices=None):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planetse") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

        Norbits=10000
        Nout = 1000

        trioprobs = np.zeros(len(trios))
        args = [Norbits, Nout, trios]
        triofeatures = self.featurefunc(sim, args) # 
        for i, trio in enumerate(trios):
            summaryfeatures = triofeatures[i] 
            if summaryfeatures['unstableinshortintegration'] == 1: # definitely unstable if unstable in short integration
                return 0
            
            if self.features is not None:
                summaryfeatures = summaryfeatures[self.features] # take subset of features used in training the model
            summaryfeatures = pd.DataFrame([summaryfeatures]) # put it in format model expects...would be nice to optimize out
            trioprobs[i] = self.model.predict_proba(summaryfeatures)[:,1][0]
        return trioprobs.min() # return minimum of 3
