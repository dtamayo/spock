from spock import features
from spock.features import get_min_secT, Trio, hillfac
import sys
import pandas as pd
import numpy as np
import rebound
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from .simsetup import setup_sim
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import os
import warnings

class FeatureClassifier:

    def __init__(self, modelfile='models/featureclassifier.json'):
        '''initializes class and imports spock model'''
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/'+modelfile)

    def predict_stable(self, sims, n_jobs = -1, Nbodytmax = 1e6, singleWrap = False):
        '''Evaluates probability of stability for a list of simulations

            Arguments:
                sims: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Nbodytmax: Max number of orbits the short integration
                        will run for. Default used to train the model is
                        1e6. Be sure to test the performance if changing this value.
                singleWrap: if a single sim is passed, whether to return the results
                            in a list of length one or not. Included for backward 
                            compatibility since original spock behaves as if
                            singleWrap = False

            return: the probability that each system is stable
        '''
        # If a list of sims is passed, they must all have the same number of 
        # particles in order to predict stability due to the way we pass to
        # XGBoost.predict_proba, this limitation does not apply for generating
        # features
        if not isinstance(sims, rebound.Simulation) \
            and len(set([sim.N_real for sim in sims])) != 1:
            raise ValueError("If running over many sims at once, "\
                             "they must have the same number of particles")

        try:
            Nplanets = sims[0].N_real-1 # we require above that all simulations have same N
        except:
            Nplanets = sims.N_real-1

        if Nplanets == 0 or Nplanets == 1:
            try:
                return np.ones(len(sims))
            except:
                return 1.

        if Nplanets == 2:
            try:
                probs = np.float64([hillfac(sim) > 1 for sim in sims])
            except:
                probs = np.float64(hillfac(sims) > 1)
            return probs

        #Generates features for each trio in each simulation
        res = self.generate_features(sims, n_jobs, Nbodytmax, singleWrap = True)

        # Separate the feature dictionaries from the bool 
        # for whether it was stable over short integration
        stable = np.array([r[1] for r in res])
        features = [r[0] for r in res]
        Nsims = len(res)


        # We take the small hit of evaluating XGBoost for all systems, 
        # and overwrite prob=0 for ones that went unstable in the 
        # short integration at the end

        # Iterates through each individual system in features list
        # for each system, iterates over each trio within,
        # for each trio, converts the generated features into a list
        # and adds that list to the np array
        featurevals = np.array([list(trio.values()) for system in features for trio in system])

        # Predicts the stability for each trio
        probs = self.model.predict_proba(featurevals)[:,1] # take 2nd column for probability it belongs to stable class

        # XGBoost evaluated a flattened list of all trios, reshape so that trios in same sim grouped
        trios_per_sim = int(len(probs)/Nsims)  # determines the number of trios per simulation
        # reshapes probabilities so that each row belongs to a system
        probs = probs.reshape((Nsims, trios_per_sim))

        # Take the minimum probability of stability within the trios 
        # for each simulation, i.e, the minimum value in each row
        probs = np.min(probs, axis=1)

        # Set probabilities for systems that went unstable within short integration to exactly zero
        probs[~stable] = 0

        # Re format depending on number of simulations and if singleWrap == False
        if Nsims == 1 and singleWrap == False:
            return probs[0]

        return probs

    def generate_features(self, sims, n_jobs=-1, Nbodytmax = 1e6, singleWrap = False):
        '''Given a simulation(s), returns features used for spock classification

            Arguments:
                sims: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Nbodytmax: Max number of orbits the short integration
                    will run for. Default used to train the model is
                    1e6. Be sure to test the performance if changing this value.
                singleWrap: if a single sim is passed, whether to return the features
                            in a list of length one or not. Included for backward 
                            compatibility since original spock behaves as if
                            singleWrap = False

            return:  returns a list of the simulations features/short term stability
        '''
        #ensures that data is in list format so everything is identical
        if isinstance(sims, rebound.Simulation):
            sims = [sims]

        if n_jobs == -1:
            n_jobs = cpu_count()
        with ThreadPool(n_jobs) as pool:
            res = pool.map((lambda sim: self._generate_features(sim, Nbodytmax)), sims)
        
        # check to unwrap data if only looking at one system and singleWrap == False
        if len(sims) == 1 and singleWrap == False:
            return res[0]

        return res
    
        

    def _generate_features(self, sim, Nbodytmax):
        # internal function that generates features for an individual simulation. User uses generate_features wrapper

        sim = setup_sim(sim) # returns initialized copy rotated to invariable plane

        # calculate Norbits for how long to run short integration and number of outputs Nout

        Norbits = get_min_secT(sim) # systems get integrated to shortest secular timescale among all trios (see SPOCK 2.0 paper)
        if Norbits > Nbodytmax:
            Norbits = Nbodytmax # failsafe to avoid cases with very long short integration times
            warnings.warn(f'Min secular timescale > Nbodytmax orbits of inner most planet '\
                          f'Defaulting to integrating to {Nbodytmax} orbits. Might affect model performance')
        # set the number of outputs in short integration to be same as in original model (80 outputs over 1e4 orbits)
        Nout = int((Norbits / 1e4) * 80)

        # make list of Trio objects for each adjacent trio
        trios = []
        for trio_indices in [[j, j+1, j+2] for j in range(1, sim.N_real - 2)]:
            trios.append(Trio(trio_indices, sim, Nout))

        # fill in features based on initial conditions
        for trio in trios: # in each trio there is two adjacent pairs
            trio.fill_starting_features(sim)

        # runs short integration and get time series for quantities we want to track
        trios, stable = self.get_tseries(sim, trios, Norbits, Nout)

        # calculate final features at end of short integration
        for trio in trios:
            trio.fill_final_features(sim)

        return [trio.features for trio in trios], stable

    def get_tseries(self, sim, trios, Norbits, Nout):
        '''Gets the time series list of data for one simulation of N bodies.

        Arguments:
            sim: simulation in question
            args: arguments in format [number of orbits,
                     number of data collections equally spaced, list of trios]

        return:
                trios: A list of Trio objects with trio.runningList being a dict of all the time series
                stable: whether or not the end configuration is stable
            '''
        # determine the smallest period that a particle in the system has
        minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
        times = np.linspace(0, Norbits*minP, Nout) # list of times to integrate to

        stable = True # stable by default and update if not
        for i, time in enumerate(times):
            # integrates each step and collects required data
            try:
                sim.integrate(time, exact_finish_time = 0)
                if  (sim._status == 4 or sim._status == 7):
                    # check for escaped particles or collisions
                    stable=False
                    return [trios, stable]
            except:
                # catch exception 
                # sim._status == 7 is checking for collisions and sim._status==4 
                # is checking for ejections
                if sim._status == 7 or sim._status == 4 :
                    # case of ejection or collision
                    stable = False
                    return [trios, stable]
                else:
                    # something else went wrong
                    raise
            for trio in trios:
                trio.fill_tseries_entry(sim, minP, i)

        # returns list of objects and whether or not stable after short integration
        return [trios, stable]

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """

        txt = r"""This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. These were done with the FeatureClassifier decision-tree model, which provides a probability of stability over $10^9$ orbits for a given input orbital configuration, derived from dynamically relevant features extracted from a short N-body integration to the system's fastest secular timescale \\citep{Thadhani_2025}."""
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

@ARTICLE{Thadhani_2025,
doi = {10.3847/2515-5172/adb150},
url = {https://dx.doi.org/10.3847/2515-5172/adb150},
year = {2025},
month = {feb},
publisher = {The American Astronomical Society},
volume = {9},
number = {2},
pages = {27},
author = {{Thadhani}, Elio and {Ba}, Yanming and {Rein}, Hanno and {Tamayo}, Daniel},
title = {SPOCK 2.0: Updates to the FeatureClassifier in the Stability of Planetary Orbital Configurations Klassifier},
journal = {Research Notes of the AAS},
}
"""
        print(txt + "\n\n\n" + bib)
