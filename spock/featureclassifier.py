import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import rebound
from xgboost import XGBClassifier

from .feature_functions import features
from .simsetup import init_sim_parameters
from .citations import cite


class FeatureClassifier():
    def __init__(self, modelfile='featureclassifier.json'):
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/models/'+modelfile)

    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_stable(self, sim, n_jobs=-1):
        """
        Predict whether passed simulation will be stable over 10^9 orbits of the innermost planet.

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test
        n_jobs (int):               Number of cores to use for calculation (only if passing more than one simulation). Default: Use all available cores. 

        Returns:

        float:  Estimated probability of stability. Will return exactly zero if configuration goes 
                unstable within first 10^4 orbits.

        """
        res = self.generate_features(sim, n_jobs=n_jobs)

        try: # separate the feature dictionaries from the bool for whether it was stable over short integration
            stable = np.array([r[1] for r in res])
            features = [r[0] for r in res]
            Nsims = len(sim)
        except:
            stable = np.array([res[1]])
            features = [res[0]]
            Nsims = 1

        # We take the small hit of evaluating XGBoost for all systems, and overwrite prob=0 for ones that went unstable in the short integration at the end
        # array of Ntrios x 10 features to evaluate with XGboost (Nsims*Ntriospersim x 10 features)
        featurevals = np.array([[val for val in trio.values()] for system in features for trio in system]) 
        probs = self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class
        # XGBoost evaluated a flattened list of all trios, reshape so that trios in same sim grouped
        trios_per_sim = int(len(probs)/Nsims)
        probs = probs.reshape((Nsims, trios_per_sim))
        # Take the minimum probability of stability within the trios for each simulation
        probs = np.min(probs, axis=1)
        # Set probabilities for systems that went unstable within short integration to exactly zero
        probs[~stable] = 0

        if Nsims == 1:
            return probs[0]
        else:
            return probs

    def generate_features(self, sim, n_jobs=-1):
        """
        Generates the set of summary features used by the feature classifier for prediction. 

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test
        n_jobs (int):               Number of cores to use for calculation (only if passing more than one simulation). Default: Use all available cores. 

        Returns:

        List of OrderedDicts:   A list of sets of features for each adjacent trio of planets in system.
                                Each set of features is an ordered dictionary of 10 summary features. See paper.
       
        stable (int):           An integer for whether the N-body integration survived the 10^4 orbits (1) or 
                                went unstable (0).
        """
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
        
        args = []
        if len(set([s.N_real for s in sim])) != 1:
            raise ValueError("If running over many sims at once, they must have the same number of particles")
        for s in sim:
            s = s.copy()
            init_sim_parameters(s)
            minP = np.min([p.P for p in s.particles[1:s.N_real]])
            self.check_errors(s)
            trios = [[j,j+1,j+2] for j in range(1,s.N_real-2)] # list of adjacent trios   
            featureargs = [10000, 80, trios]
            args.append([s, featureargs])

        def run(params):
            sim, featureargs = params
            triofeatures, stable = features(sim, featureargs)
            return triofeatures, stable

        if len(args) == 1: # single sim
            res = run(args[0])    # stable will be 0 if an orbit is hyperbolic
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            with ThreadPool(n_jobs) as pool:
                res = pool.map(run, args)
        return res

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """
        
        txt = """This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. These were done with the FeatureClassifier decision-tree model, which provides a probability of stability over $10^9$ orbits for a given input orbital configuration, derived from dynamically relevant features extracted from short $10^4$-orbit N-body integrations \\citep[see also][]{spockI}."""
        bib = """
@ARTICLE{spock,
   author = {{Tamayo}, Daniel and {Cranmer}, Miles and {Hadden}, Samuel and {Rein}, Hanno and {Battaglia}, Peter and {Obertas}, Alysa and {Armitage}, Philip J. and {Ho}, Shirley and {Spergel}, David N. and {Gilbertson}, Christian and {Hussain}, Naireen and {Silburt}, Ari and {Jontof-Hutter}, Daniel and {Menou}, Kristen},
    title = "{Predicting the long-term stability of compact multiplanet systems}",
  journal = {Proceedings of the National Academy of Science},
 keywords = {machine learning, dynamical systems, UAT:498, orbital dynamics, UAT:222, Astrophysics - Earth and Planetary Astrophysics},
     year = 2020,
    month = aug,
   volume = {117},
   number = {31},
    pages = {18194-18205},
      doi = {10.1073/pnas.2001258117},
archivePrefix = {arXiv},
   eprint = {2007.06521},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2020PNAS..11718194T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}

@ARTICLE{spockI,
   author = {{Tamayo}, Daniel and {Silburt}, Ari and {Valencia}, Diana and {Menou}, Kristen and {Ali-Dib}, Mohamad and {Petrovich}, Cristobal and {Huang}, Chelsea X. and {Rein}, Hanno and {van Laerhoven}, Christa and {Paradise}, Adiv and {Obertas}, Alysa and {Murray}, Norman},
    title = "{A Machine Learns to Predict the Stability of Tightly Packed Planetary Systems}",
  journal = {\apjl},
 keywords = {celestial mechanics, chaos, planets and satellites: dynamical evolution and stability, Astrophysics - Earth and Planetary Astrophysics},
     year = 2016,
    month = dec,
   volume = {832},
   number = {2},
      eid = {L22},
    pages = {L22},
      doi = {10.3847/2041-8205/832/2/L22},
archivePrefix = {arXiv},
   eprint = {1610.05359},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2016ApJ...832L..22T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
        print(txt + "\n\n\n" + bib)
