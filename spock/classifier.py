import dill
import pandas as pd
import os
import numpy as np
import os
from xgboost import XGBClassifier
from . import feature_functions

class spockClassifier():
    def __init__(self):
        pwd = os.path.dirname(__file__)
        self.model, self.features, self.featureargs = dill.load(open("../models/spock.pkl", "rb"))
        self.featurefunc = getattr(feature_functions, 'features')
        #self.model = XGBClassifier()
        #self.model.load_model(pwd+'/../models/spocknoAMD2.bin')

    def predict(self, sim, indices=None):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planetse") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios
        
        featureargs = [f for f in self.featureargs] + [trios]
        trioprobs = np.zeros(len(trios))
        triofeatures, stable = self.featurefunc(sim, featureargs) # 
        if not stable:
            return 0

        for i, trio in enumerate(trios):
            summaryfeatures = triofeatures[i] 
            summaryfeatures = pd.DataFrame([summaryfeatures[self.features]])# put it in format model expects...would be nice to optimize out
            trioprobs[i] = self.model.predict_proba(summaryfeatures)[:,1] # probability of stability
        return trioprobs.min() # return minimum of 3
