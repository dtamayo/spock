#from spock import simsetup
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
from .simsetup import init_sim_parameters
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
# from multiprocessing.pool import ThreadPool



import os


class FeatureClassifier:


    def __init__(self, modelfile='models/spock.bin'):
        '''initializes class and imports spock model'''
        pwd = os.path.dirname(__file__)
        self.model = XGBClassifier()
        self.model.load_model(pwd + '/'+modelfile)



    def predict_stable(self,sim, n_jobs = -1, Tmax = False):
        '''runs spock classification on a list of simulations
            Arguments: sim --> simulation or list of simulations
                    n_jobs --> number of jobs you want to run with multi processing
                    Tmax --> whether you want to run simulation to Tmax (5* secular time scale from Yang and Tamayo), and at least to 1e4, or all to 1e4
            return: the probability that a system is stable
            '''

        #specify features
        near = ['EMcrossnear', 'EMfracstdnear', 'EPstdnear', 'MMRstrengthnear']
        far = ['EMcrossfar', 'EMfracstdfar', 'EPstdfar', 'MMRstrengthfar']
        megno = ['MEGNO', 'MEGNOstd']

        features = near + far + megno
#

        simFeatureList = self.simToData(sim, n_jobs, Tmax)
        results = []
        for s in simFeatureList:
            if s[1]==False:
                results.append(0)
            else:
                eachTrio = []
                for eachT in s[0]:
                    eachTrio.append(self.model.predict_proba(pd.DataFrame.from_dict(eachT, orient="index").T[features])[:,1][0]) 
                results.append(min(eachTrio))

        if len(results)==1:
            return results[0]
        else:
            return results
    
    def generate_features(self, sim, n_jobs = -1, Tmax = False):
        '''helper function to fit spock syntex standard'''
        data = self.simToData(sim,n_jobs, Tmax)
        #nicely wraps data if only evaluating one system
        if len(data)==1:
            return data[0]
        else:
            return data


    
    def simToData(self, sim,n_jobs, Tmax):
        '''given a simulation, or list of simulations, returns data required for spock clasification.
        
            Arguments: sim --> simulation or list of simulations
                n_jobs --> number of jobs you want to run with multi processing
                Tmax --> whether you want to run simulation to Tmax (5* secular time scale from Yang and Tamayo), and at least to 1e4, or all to 1e4
            
            return:  returns a list of the simulations features/short term stability'''
        

        

        if isinstance(sim, rebound.Simulation):
            sim = [sim]
            
        
        if len(set([s.N_real for s in sim])) != 1:
            raise ValueError("If running over many sims at once, they must have the same number of particles")
        
        
        if len(sim)==1:
            return [self.run(sim[0], Tmax)]
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            with ThreadPool(n_jobs) as pool:
                res = pool.map(self.run, sim, Tmax)
            
            return res
        
    

    def run(self, s, Tmax):
        Norbits = 1e4 #number of orbits for short intigration, usually 10000
        Nout = 80 #number of data collections spaced throughought, usually 80
        if float(rebound.__version__[0])>=4:
            #check for rebound version here, if its version 4 or later then sim.copy() should be supported, if a old version of rebound,
            #we will change the simulation in place
            s = s.copy() #creates a copy as to not alter simulation
        init_sim_parameters(s) #initializes the simulation
        self.check_errors(s) #checks for errors
        
        trios = [[j,j+1,j+2] for j in range(1,s.N_real-2)] # list of adjacent trios
        #check if we want to use Tmax option
        if Tmax == True:
            maxList = []
            for each in trios:
                maxList.append(self.getsecT(s,each))
            intT = 5 * max(maxList)
            if intT>1e4:
                Norbits = intT
                Nout = int((Norbits/1e4)*80)
            
    
        featureargs = [Norbits, Nout, trios] #featureargs is: [number of orbits, number of stops, set of trios]
        return self.runSim(s,featureargs) #adds data to results. calls runSim helper function which returns the data list for sim

    
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
        

    def getsecT(self,sim, trio):
        '''calculates the secular time scale for a given trio in a simulation in accordance to yang and tamayo'''
        ps = sim.particles
        p1, p2, p3 = ps[trio[0]], ps [trio[1]], ps[trio[2]]
        Tmax = (ps[0].m/(p1.m+p2.m+p3.m))*((1-(p1.a/p3.a))**2)*p3.P/4
        return Tmax