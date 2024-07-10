import rebound
import numpy as np
from collections import OrderedDict
import warnings
import pandas as pd
import features

global featureCount
featureCount = 10

def get_pairs(sim, trio):
    '''returns the three pairs of the given trio.
    
    Arguments:
        sim: simulation in question
        trio: indicies of the three particles in question, formated as [p1,p2,p3]
    return:
        return: returns the two pairs in question, formated as [[near pair, index, index], [far pair, index, index]]'''
    #print(trio)
    ps = sim.particles
    sortedindices = sorted(trio, key=lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = (ps[sortedindices[1]].a-ps[sortedindices[0]].a)/ps[sortedindices[0]].a
    EMcrossOuter = (ps[sortedindices[2]].a-ps[sortedindices[1]].a)/ps[sortedindices[1]].a
    
    

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]], ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]], ['far', sortedindices[0], sortedindices[1]]]
    


def get_tseries(sim, args):
    '''gets the time series list of data for one simulation of N bodies.
    
    Arguments:
        sim: simulation in question
        args: arguments in format [number of orbits, number of data collections equally spaced, list of trios]'''
    Norbits = args[0] #number of orbits
    Nout = args[1] #number of data collection
    trios = args[2] #list of each planet set trio

    #a10s = [sim.particles[trio[0]].a for trio in trios] #collects a list of the closest particle in each trio
    minP = np.min([p.P for p in sim.particles[1:sim.N_real]])

    #minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]]) #determins the smallest period that a particle has

    #not sure what to do to make sure hyperbolic case runs
    
    times = np.linspace(0, Norbits*minP, Nout) #list of times to intigrate to

    #triopairs = []
    triotseries: list[features.Trio] =[]
    #forms the list that will later consist of each trio pair, and the tseries for each list
    #print(trios)
    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs 
        #fills triopairs with each pair, and fills triotseries with the Trio class 
        #eachpair = get_pairs(sim, trio)
        #triopairs.append(eachpair)
        triotseries.append(features.Trio())
        triotseries[tr].fillVal(Nout)
        #print(trio)
        triotseries[tr].startingFeatures(sim, get_pairs(sim,trio)) #puts in the valeus that depend on initial cond
        #triotseries will be a list of the objects that have all the data and Trio
    
    fail = False
    for i, time in enumerate(times):
        #intigrates each step and collects required data
        try:
            sim.integrate(time, exact_finish_time=0)
        except rebound.Collision:
            #if not stable after intigration jump, just ends simulation
            fail = True
            #returns list of objects and whether or not stable after short intigration
            #return triotseries, stable
            break

        
        
        for tr, trio in enumerate(trios):
            #populates data for each trio
            #print(trio)
            triotseries[tr].populateData( sim, trio, get_pairs(sim,trio), minP,i)
    
    if fail == False:
        stable = True
    else: 
        stable = False
        fail = False
    #returns list of objects and whether or not stable after short intigration
    return [triotseries, stable]






