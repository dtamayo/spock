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
        args: arguments in format [number of orbits, number of data collections equally spaced, list of trios]'''
    Norbits = args[0] #number of orbits
    Nout = args[1] #number of data collection
    trios = args[2] #list of each planet set trio

    minP = np.min([p.P for p in sim.particles[1:sim.N_real]])#determins the smallest period that a particle in the system has


    times = np.linspace(0, Norbits*np.abs(minP), Nout) #list of times to intigrate to

    triotseries: list[features.Trio] =[]
    #forms the list that will later consist of each trio pair, and the tseries for each list
    
    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs 
        #fills triopairs with each pair, and fills triotseries with the Trio class 
        triotseries.append(features.Trio(trio, sim))
        triotseries[tr].fillVal(Nout)
        triotseries[tr].startingFeatures(sim) #puts in the valeus that depend on initial cond
        #triotseries will be a list of the objects that have all the data and Trio
    
    fail = False
    for i, time in enumerate(times):
        #intigrates each step and collects required data
        try:
            sim.integrate(time, exact_finish_time=0)
            if float(rebound.__version__[0])<=4 and (sim._status==4 or sim._status==5):
                #break condition for old version of rebound
                fail = True
                break
        except():
            #catch exception 
            #sim._status==5 is checking for collisions and sim._status==4 is checking for exceptions
            if sim._status==5 or sim._status==4:
                #case of ejection or collision
                fail = True
                break
            else:
                #something else went wrong
                raise
        for tr, trio in enumerate(trios):
            #populates data for each trio
            triotseries[tr].populateData( sim, minP,i)
    
    if fail == False:
        stable = True
    else: 
        stable = False
        fail = False
    #returns list of objects and whether or not stable after short intigration
    return [triotseries, stable]






