from spock import simsetup
from spock import features
from spock import ClassifierSeries
# from features import *
# from ClassifierSeries import *
# from simsetup import *
import sys
import pandas as pd
import numpy as np
import rebound
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os


class FeatureKlassifier:


    def __init__(self, modelfile='models/SPOCKalt.bin'):
        '''initializes class and imports spock model'''
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/'+modelfile)



    def predict_stable(self,sim):
        '''runs spock classification on a list of simulations'''
        simFeatureList = self.simToData(sim)
        results = []
        for s in simFeatureList:
            if s[1]==False:
                results.append(0)
            else:
                eachTrio = []
                for eachT in s[0]:
                    eachTrio.append(self.model.predict_proba(pd.DataFrame.from_dict(eachT, orient="index").T)[:,0][0]) 
                results.append(min(eachTrio))

        if len(results)==1:
            return results[0]
        else:
            return results
    
    def generate_features(self, sim):
        '''helper function to fit spock syntex standard'''
        data = self.simToData(sim)
        if len(data)==1:
            return data[0]
        else:
            return data


    
    def simToData(self, sim):
        '''given a simulation, or list of simulations, returns data required for spock clasification.
        
            Arguments: sim --> simulation or list of simulations
            
            return:  returns a list of the simulations features/short term stability'''
        #tseries, stable = get_tseries(sim, args)

        Norbits = 1e4 #number of orbits for short intigration, usually 10000
        Nout = 80 #number of data collections spaced throughought, usually 80

        if isinstance(sim, rebound.Simulation):
            sim = [sim]
            
        #args = []
        if len(set([s.N_real for s in sim])) != 1:
            raise ValueError("If running over many sims at once, they must have the same number of particles")
        
        results = [] #results of the intigrations for each, if only one simulation return will not be in a list

        #for intigrated systems
        for s in sim:
            s = s.copy() #creates a copy as to not alter simulation
            simsetup.init_sim_parameters(s) #initializes the simulation
            self.check_errors(s) #checks for errors
            trios = [[j,j+1,j+2] for j in range(1,s.N_real-2)] # list of adjacent trios   
            featureargs = [Norbits, Nout, trios] #featureargs is: [number of orbits, number of stops, set of trios]
            
            results.append(self.runSim(s,featureargs)) #adds data to results. calls runSim helper function which returns the data list for sim
        
        return results
    
    def runSim(self, sim, args):
        '''returns the data list of features for a given simulation
            
            Arguments: 
                sim: simulation in question
                args: contains number or orbits, number of data collections, and the set of all trios
                
            return: returns data list, which, contains the set of features for each trio, and whether sys stable in short intigration
        
        '''

        triotseries, stable = ClassifierSeries.get_tseries(sim, args) #runs the intigration on the simulation, and returns the filled objects for each trio and short stability
        #calculate final vals

       
        dataList = []
        for each in triotseries:
            each.fill_features(args) #turns runningList data into final features
            dataList.append(each.features) #appends each feature results to dataList
        #dataList.append(stable) #adds short term stability
        return dataList, stable

    def check_errors(self, sim):
        '''ensures enough planets/stars for spock to run'''
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets")