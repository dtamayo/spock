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

    def check_errors(self, sim, trios):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
        for trio in trios:
            try:
                if len(trio) != 3:
                    raise
            except:
                raise AttributeError("SPOCK Error: trios must be a list of trios of particle indices, e.g., [[1,2,3]], [1,2,4]], or [[1,2,3]]")
            
            if min(trio) <= 0:
                raise AttributeError("SPOCK Error: indices cannot include primary (index 0).")
            if max(trio) >= sim.N_real:    
                raise AttributeError("SPOCK Error: indices cannot include indices larger than number of bodies in simulation.")
    #if not isinstance(i1, int) or not isinstance(i2, int) or not isinstance(i3, int):
        
        #raise AttributeError("SPOCK  Error: Particle indices passed to spock_features were not integers")

    def predict_stable(self, sim, dtfrac=0.05, trios=None):

        # Flag hyperbolic initial conditions as unstable before
        # using orbital period to set timestep of short integration
        # This is an acceptable init. cond. so don't raise exception
        if check_hyperbolic(sim) == True: 
            return 0

        sim = sim.copy()
        check_valid_sim(sim)
        init_sim_parameters(sim, dtfrac)

        # check hyperbolic Should return 0 for stable if True, rather than raise

        triofeatures, collision = self.generate_features(sim, trios, copysim=False) # already copied
        if collision == True or check_hyperbolic(sim) == True:
            return 0
       
        trioprobs = self.predict_from_features(triofeatures)

        return trioprobs.min() # return minimum of probability amoong all trios tested

    def generate_features(self, sim, trios=None, copysim=True):
        if copysim is True:
            sim = sim.copy()

        if trios is None:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

        self.check_errors(sim, trios)

        featureargs = [10000, 80, trios]
        triofeatures, stable = features(sim, featureargs) 
        return triofeatures, stable

    def predict_from_features(self, featureslist):
        # xgboost model expects a 2D array of shape (Npred, Nfeatures) where Npred is number of samples to predict, Nfeatures is # of features per sample
        featurevals = np.array([[val for val in features.values()] for features in featureslist])
        return self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class


