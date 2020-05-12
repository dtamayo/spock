import numpy as np
import os
from xgboost import XGBClassifier
from .feature_functions import features
from .simsetup import set_sim_parameters

class StabilityClassifier():
    def __init__(self, modelfile='spock_new.json', features=None):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)
        self._features = features

    def predict(self, sim, indices=None, copy=True):
        if copy:
            sim = sim.copy()
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

        set_sim_parameters(sim)
        featureargs = [10000, 80] + [trios]
        trioprobs = np.zeros(len(trios))
        triofeatures = features(sim, featureargs) # 
        if triofeatures[0]['stable_in_short_integration'] == False:
            return 0

        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # f / sample
        if self._features:
            triofeaturevals = np.array([[feat[f] for f in self._features] for feat in triofeatures])
        else:
            triofeaturevals = np.array([[val for val in feat.values()] for feat in triofeatures])
        trioprobs = self.model.predict_proba(triofeaturevals)[:,1] # take 2nd column for probability it belongs to stable class
        
        return trioprobs.min() # return minimum of 3
