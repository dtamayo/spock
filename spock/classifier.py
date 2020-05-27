import numpy as np
import os
from xgboost import XGBClassifier
from .feature_functions import features
from .simsetup import set_sim_parameters, rescale, set_timestep, init_sim

class StabilityClassifier():
    def __init__(self, modelfile='spock.json', features=None):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)
        self._features = features

    def predict(self, sim, dtfrac=0.05, indices=None, archive_filename=None, archive_interval=None):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
       
        sim = sim.copy()
        sim = init_sim(sim, dtfrac, archive_filename=archive_filename, interval=interval)

        triofeatures, stable = self.generate_features(sim, indices, archive_filename, archive_interval)
        if not stable:
            return 0

        
        return self._trioprobs.min() # return minimum of 3

    def generate_features(self, sim, indices=None, archive_filename=None, archive_interval=None):
        if indices:
            if len(indices) != 3
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

        featureargs = [10000, 80, trios]
        triofeatures, stable = features(self._sim, featureargs) # 
        return triofeatures, stable

    def predict_from_features(self, featurelist):
        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # of features per sample
        featurevals = np.array([[val for val in features.values()] for features in featureslist])
        return self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class


