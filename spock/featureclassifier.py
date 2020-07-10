import numpy as np
import os
from xgboost import XGBClassifier
from .feature_functions import features
from .simsetup import init_sim_parameters, check_valid_sim, check_hyperbolic

class FeatureClassifier():
    def __init__(self, modelfile='featureclassifier.json'):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)

    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_stable(self, sim):
        """
        Predict whether passed simulation will be stable over 10^9 orbits of the innermost planet.

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test

        Returns:

        float:  Estimated probability of stability. Will return exactly zero if configuration goes 
                unstable within first 10^4 orbits.

        """
        triofeatures, stable = self.generate_features(sim)
        if stable == False:
            return 0
       
        trioprobs = self.predict_from_features(triofeatures)
        return trioprobs.min()          # minimum prob among all trios tested

    def generate_features(self, sim):
        """
        Generates the set of summary features used by the feature classifier for prediction. 

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test

        Returns:

        List of OrderedDicts:   A list of sets of features for each adjacent trio of planets in system.
                                Each set of features is an ordered dictionary of 10 summary features. See paper.
       
        stable (int):           An integer for whether the N-body integration survived the 10^4 orbits (1) or 
                                went unstable (0).
        """
        sim = sim.copy()
        init_sim_parameters(sim)
        self.check_errors(sim)
        
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios   
        featureargs = [10000, 80, trios]
        triofeatures, stable = features(sim, featureargs)    # stable will be 0 if an orbit is hyperbolic
                                                             # sim.dt = nan in init_sim_parameters
        
        return triofeatures, stable

    def predict_from_features(self, triofeatures):
        """
        Estimate probability of stability from the list of features created by FeatureClassifier.generate_features.

        Parameters:

        triofeatures (List of Ordered Dicts):   Sets of features for each adjacent planet trio
                                                (returned from FeatureClassifier.generate_features)

        Returns:

        list (float): Estimated probabilities of stability for set of features passed (for each adjacent trio of planets).
        """

        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # of features per sample
        featurevals = np.array([[val for val in features.values()] for features in triofeatures])
        return self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class
