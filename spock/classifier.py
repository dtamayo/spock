import numpy as np
import os
from xgboost import XGBClassifier
from .feature_functions import features
from .simsetup import init_sim_parameters, check_valid_sim, check_hyperbolic

class StabilityClassifier():
    def __init__(self, modelfile='spock.json'):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)

    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_stable(self, sim):
        triofeatures, stable = self.generate_features(sim)
        if stable == False:
            return 0
       
        trioprobs = self.predict_from_features(triofeatures)
        return trioprobs.min()          # minimum prob among all trios tested

    def generate_features(self, sim):
        sim = sim.copy()
        init_sim_parameters(sim)
        self.check_errors(sim)
        
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios   
        featureargs = [10000, 80, trios]
        triofeatures, stable = features(sim, featureargs)
        
        return triofeatures, stable

    def predict_from_features(self, triofeatures):
        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # of features per sample
        featurevals = np.array([[val for val in features.values()] for features in triofeatures])
        return self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class
