from spock import features
from spock import ClassifierSeries
import sys
import pandas as pd
import numpy as np
import rebound
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from .simsetup import init_sim_parameters
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

    def predict_stable(self,sim, n_jobs = -1, Nbodytmax = 1e6):
        '''Evaluates probability of stability for a list of simulations

            Arguments: 
                sim: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Nbodytmax: Max number of orbits the short integration
                        will run for. Default used to train the model is
                        1e6. Be sure to test the performance if changing this value.

            return: the probability that each system is stable
        '''
        # If a list of sims is passed, they must all have the same number of 
        # particles in order to predict stability due to the way we pass to
        # XGBoost.predict_proba, this limitation does not apply for generating
        # features
        if not isinstance(sim, rebound.Simulation) \
            and len(set([s.N_real for s in sim])) != 1:
            raise ValueError("If running over many sims at once, "\
                             "they must have the same number of particles")

        #Generates features for each trio in each simulation
        res = self.simToData(sim, n_jobs, Nbodytmax)

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

        # Re format depending on number of simulations
        if Nsims == 1:
            return probs[0]
        else:
            return probs
    
    def generate_features(self, sim, n_jobs = -1, Nbodytmax = 1e6):
        '''helper function to fit spock syntax standard
            Arguments:
                    sim: simulation or list of simulations
                    n_jobs: number of jobs to run with multi processing
                    Nbodytmax: Max number of orbits the short integration
                        will run for. Default used to train the model is
                        1e6. Be sure to test the performance if changing this value.
    
            return: features for given system or list of systems
        '''
        data = self.simToData(sim, n_jobs, Nbodytmax)
        # Nicely wraps data if only evaluating one system
        if len(data) == 1:
            return data[0]
        else:
            return data

    def simToData(self, sim, n_jobs, Nbodytmax = 1e6):
        '''Given a simulation(s), returns data required for spock classification
        
            Arguments:
                sim: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Nbodytmax: Max number of orbits the short integration
                    will run for. Default used to train the model is
                    1e6. Be sure to test the performance if changing this value.
            
            return:  returns a list of the simulations features/short term stability
        '''
        #ensures that data is in list format so everything is identical
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
                
        if len(sim)==1:
            #retuns the data for a single simulation
            return [self.run(sim[0], Nbodytmax)]
        else:
            #if more then one sim is passed, uses thread pooling
            #to generate data for each
            if n_jobs == -1:
                n_jobs = cpu_count()
            with ThreadPool(n_jobs) as pool:
                res = pool.map((lambda s : self.run(s, Nbodytmax)), sim)
            return res

    def run(self, s, Nbodytmax):
        '''Sets up simulation and starts data collection

        Arguments:
            s: The simulation you would like to generate data for
        
        return: data list for sim and whether or not it is stable in tuple
        '''
        TIMES_TSEC = 1 #all systems get integrated to 1 secular time scale
        
        if float(rebound.__version__[0]) >= 4:
            #check for rebound version here, if version 4 or later then
            # sim.copy() should be supported, if old version of rebound,
            # we will change the simulation in place
            s = s.copy() #creates a copy as to not alter simulation
        else:
            warnings.warn('Due to old REBOUND, SPOCK will change sim in place')

        init_sim_parameters(s) #initializes the simulation
        self.check_errors(s) #checks for errors
        
        trios = [[j, j+1, j+2] for j in range(1, s.N_real - 2)] # list of adjacent trios

        minList = []
        for each in trios:
            minList.append(ClassifierSeries.getsecT(s, each)) # gets secular time
        intT = TIMES_TSEC * min(minList)# finds the trio with shortest time scale
        if intT > Nbodytmax:
            intT = Nbodytmax # check to make sure time scale is not way to long
            warnings.warn(f'Sim Tsec > {Nbodytmax} orbits of inner most planet '\
                          f'thus, system will only be integrated to {Nbodytmax} orbits. '\
                            'This might affect model performance')
            # if it is, default to Nbodytmax, very few systems should have this issue
            # if Nbodytmax >=1e6
        Norbits = intT # set the number of orbits to be equal to Tsec
        # set the number of data collections to be equally spaced with same
        # spacing as old spock, 80 data collections every 1e4 orbits, scaled
        Nout = int((Norbits / 1e4) * 80)
            
            
        # featureargs is: [number of orbits, number of stops, set of trios]
        featureargs = [Norbits, Nout, trios] 
        # adds data to results. 
        # calls runSim helper function which returns the data list for sim
        return self.runSim(s, featureargs) 

    
    def runSim(self, sim, args):
        '''returns the data list of features for a given simulation
            
            Arguments: 
                sim: simulation in question
                args: contains number or orbits, number of data collections, 
                    and the set of all trios
                
            return: returns list containing the set of features for each trio,
                    and whether sys stable in short integration
        '''
        # runs the intigration on the simulation, 
        # and returns the filled objects for each trio and short stability
        triotseries, stable = ClassifierSeries.get_tseries(sim, args) 
        # calculate final vals
        dataList = []
        for each in triotseries:
            each.fill_features(args) # turns runningList data into final features
            dataList.append(each.features) # appends each feature results to dataList
        return dataList, stable

    def check_errors(self, sim):
        '''ensures enough planets/stars for spock to run'''
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets")

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """
        
        txt = """This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. These were done with the FeatureClassifier decision-tree model, which provides a probability of stability over $10^9$ orbits for a given input orbital configuration, derived from dynamically relevant features extracted from short N-body integrations to a systems secular timescale \\citep[see also][]{Thadhani_2025, spockI}."""
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
  journal = {\\apjl},
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
