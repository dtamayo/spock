import rebound
import numpy as np
from collections import OrderedDict
import warnings
import pandas as pd
from spock import features






def get_tseries(sim, args):
    '''gets the time series list of data for one simulation of N bodies.
    
    Arguments:
        sim: simulation in question
        args: arguments in format [number of orbits,
                 number of data collections equally spaced, list of trios]
        
    return: 
            triotseries: The time series of collected data for each trio
            stable: whether or not the end configuration is stable
        '''
    Norbits = args[0] #number of orbits
    Nout = args[1] #number of data collection
    trios = args[2] #list of each planet set trio
    # determine the smallest period that a particle in the system has
    minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])


    times = np.linspace(0, Norbits*minP, Nout) #list of times to intigrate to

    triotseries: list[features.Trio] =[]
    #forms the list that will later consist of each trio pair, 
    # and the tseries for each list
    
    for tr, trio in enumerate(trios): #in each trio there is two adjacent pairs 
        #fills triopairs with each pair, 
        # and fills triotseries with the Trio class 
        triotseries.append(features.Trio(trio, sim))
        triotseries[tr].fillVal(Nout)
        #puts in the valeus that depend on initial cond
        triotseries[tr].startingFeatures(sim) 
        #triotseries will be a list of the obj that have all the data and Trio
    
    stable = True
    for i, time in enumerate(times):
        #intigrates each step and collects required data
        try:
            
            sim.integrate(time, exact_finish_time=0)
            if  (sim._status==4 or sim._status==7):
                #check for escaped particles or collisions
                stable=False
                return [triotseries, stable]
        except:
            #catch exception 
            #sim._status==7 is checking for collisions and sim._status==4 
            # is checking for ejections
            if sim._status==7 or sim._status==4 :
                #case of ejection or collision
                stable=False
                return [triotseries, stable]
            else:
                #something else went wrong
                raise
        for tr, trio in enumerate(trios):
            #populates data for each trio
            triotseries[tr].populateData( sim, minP,i)
    #returns list of objects and whether or not stable after short integration
    return [triotseries, stable]


def getsecT(sim, trio):
    '''calculates the secular time scale for a given trio in a simulation
    
        Arguments:
            sim: the simulation that contains the trio who's 
                secular time scale you want
            trio: the trio who's secular timescale you want, 
                Note: should be a list of body indexes

        Note: This is done in accordance to Yang & Tamayo 2024
    '''
    ps = sim.particles
    p1 ,p2,p3 = ps[trio[0]], ps[trio[1]], ps[trio[2]]
    #determins the smallest period that a particle in the system has
    minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
    m1 = p1.m
    m2 = p2.m
    m3 = p3.m
    m_tot = m1+m2+m3
    mu1, mu3 = m1/m_tot, m3/m_tot
    alpha12, alpha23  = p1.a/p2.a, p2.a/p3.a
    ec12 = alpha12**(-1/4)*alpha23**(3/4)*alpha23**(-1/8)*(1-alpha12)
    ec23 = alpha23**(-1/2)*alpha12**(1/8)*(1-alpha23)
    w1 = np.abs(p3.n/(2*np.pi)*m_tot*(mu1/(mu1+mu3)
                                      /ec12**2+mu3/(mu1+mu3)/ec23**2))
    Tsec = 2*np.pi/w1
    #normalize secular timescale to be in terms of 
    # number of orbits of inner most planet
    return Tsec/minP



