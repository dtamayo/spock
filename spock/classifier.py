import numpy as np
import os
from xgboost import XGBClassifier
from .feature_functions import features
from .simsetup import set_sim_parameters, rescale, set_timestep

class StabilityClassifier():
    def __init__(self, modelfile='spock.json', features=None):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)
        self._features = features

    def predict(self, sim, indices=None, manual=False):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios
        
        if manual == False:
            self._sim = self.init_sim(sim)

        featureargs = [10000, 80] + [trios]
        trioprobs = np.zeros(len(trios))
        self._triofeatures, stable = features(self._sim, featureargs) # 
        if not stable:
            return 0

        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # f / sample
        triofeaturevals = np.array([[val for val in feat.values()] for feat in self._triofeatures])
        self._trioprobs = self.model.predict_proba(triofeaturevals)[:,1] # take 2nd column for probability it belongs to stable class
        
        return self._trioprobs.min() # return minimum of 3

    def init_sim(self, sim):
        sim = rescale(sim)
        set_sim_parameters(sim)
        set_timestep(sim, dtfrac=0.06)

        return sim

