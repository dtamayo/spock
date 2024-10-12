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

    def __init__(self, modelfile='models/spock.bin'):
        '''initializes class and imports spock model'''
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/'+modelfile)

    def predict_stable(self,sim, n_jobs = -1, Tmax = False):
        '''runs spock classification on a list of simulations

            Arguments: 
                sim: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Tmax: whether you want to run simulation to Tmax 
                    (5* secular time scale from Yang and Tamayo), 
                    and at least to 1e4, or all to 1e4

            return: the probability that each ystem is stable
        '''

        #Generates features for each trio in each simulation
        simFeatureList = self.simToData(sim, n_jobs, Tmax)

        results = []
        for s in simFeatureList:
            #loops through each simulation
            if s[1]==False:
                #If the system goes unstable during the intigration
                #returns a 0% chance that it is stable
                results.append(0)
            else:
                #for each simulation, predicts probability for each trio,
                # and then returns the lowest probability

                #creates a array of lists, each of which contain features for
                # a specific trio in the simulation. Features are in a
                # dictionary thus lambda function is used to get values.
                # We are excluding the last element of each set of features
                # since that is the feature that passes secular timescale
                # which is not implimented in model yet
                eachTrio = np.array(list(map(lambda L: list(L.values())[:-1],s[0])))
                
                #predict the probability for each trio in the sim
                probability = self.model.predict_proba(eachTrio)
                #add the lowest probability to results list
                results.append(min(probability[:,1]))
        
        if len(results)==1:
            #if only dealing with one sim, does not return as a list of results
            #per old spock standard
            return results[0]
        else:
            return np.array(results)
    
    def generate_features(self, sim, n_jobs = -1, Tmax = False):
        '''helper function to fit spock syntax standard
            Arguments:
                    sim: simulation or list of simulations
                    n_jobs: number of jobs to run with multi processing
                    Tmax: whether you want to run simulation to Tmax 
                        (5* secular time scale from Yang and Tamayo), 
                        and at least to 1e4, or all to 1e4
        '''
        data = self.simToData(sim,n_jobs, Tmax)
        #nicely wraps data if only evaluating one system
        if len(data)==1:
            return data[0]
        else:
            return data

    def simToData(self, sim, n_jobs, Tmax):
        '''given a simulation(s), returns data required for spock clasification
        
            Arguments:
                sim: simulation or list of simulations
                n_jobs: number of jobs you want to run with multi processing
                Tmax: whether you want to run simulation to Tmax 
                    (5* secular time scale from Yang and Tamayo), 
                    and at least to 1e4, or all to 1e4
            
            return:  returns a list of the simulations features/short term stability
        '''
        #ensures that data is in list format so everything is identical
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
                
        if len(sim)==1:
            #retuns the data for a single simulation
            return [self.run(sim[0], Tmax)]
        else:
            #if more then one sim is passed, uses thread pooling
            #to generate data for each
            if n_jobs == -1:
                n_jobs = cpu_count()
            with ThreadPool(n_jobs) as pool:
                res = pool.map(lambda s: self.run(s, Tmax), sim)
            return res

    def run(self, s, Tmax):
        '''
        Sets up simulation and starts data collection

        Arguments:
            s: The simulation you would like to generate data for
            Tmax: Whether or not you want to run to Tmax
        '''
        TIMES_TSEC = 1
        Norbits = 1e4 #number of orbits for short intigration, usually 10000
        Nout = 80 #number of data collections spaced throughought, usually 80
        if float(rebound.__version__[0])>=4:
            #check for rebound version here, if version 4 or later then
            # sim.copy() should be supported, if old version of rebound,
            # we will change the simulation in place
            s = s.copy() #creates a copy as to not alter simulation
        else:
            warnings.warn('Due to old REBOUND, SPOCK will change sim in place')

        init_sim_parameters(s) #initializes the simulation
        self.check_errors(s) #checks for errors
        
        trios = [[j,j+1,j+2] for j in range(1,s.N_real-2)] # list of adjacent trios
        #check if we want to use Tmax option
        if Tmax == True:
            maxList = []
            for each in trios:
                maxList.append(ClassifierSeries.getsecT(s,each))
            intT = TIMES_TSEC * max(maxList)
            if intT>1e5:
                intT = 1e5
            Norbits = intT
            Nout = int((Norbits/1e4)*80)
            
            
        #featureargs is: [number of orbits, number of stops, set of trios]
        featureargs = [Norbits, Nout, trios] 
        #adds data to results. 
        #calls runSim helper function which returns the data list for sim
        return self.runSim(s,featureargs) 

    
    def runSim(self, sim, args):
        '''returns the data list of features for a given simulation
            
            Arguments: 
                sim: simulation in question
                args: contains number or orbits, number of data collections, 
                    and the set of all trios
                
            return: returns list containing the set of features for each trio,
                    and whether sys stable in short intigration
        '''
        #runs the intigration on the simulation, 
        #and returns the filled objects for each trio and short stability
        triotseries, stable = ClassifierSeries.get_tseries(sim, args) 
        #calculate final vals
        dataList = []
        for each in triotseries:
            each.fill_features(args) #turns runningList data into final features
            dataList.append(each.features) #appends each feature results to dataList
        return dataList, stable

    def check_errors(self, sim):
        '''ensures enough planets/stars for spock to run'''
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets")
        
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
