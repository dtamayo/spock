import rebound
import numpy as np
import itertools
from scipy.optimize import brenth
from collections import OrderedDict
import warnings
import pandas as pd
from .feature_functions import find_strongest_MMR

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis
profile = lambda _: _

def get_pairs(sim, indices):
    ps = sim.particles
    sortedindices = sorted(indices, key=lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = (ps[sortedindices[1]].a-ps[sortedindices[0]].a)/ps[sortedindices[0]].a
    EMcrossOuter = (ps[sortedindices[2]].a-ps[sortedindices[1]].a)/ps[sortedindices[1]].a

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]], ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]], ['far', sortedindices[0], sortedindices[1]]]

@profile
def populate_extended_trio(sim, trio, pairs, tseries, i, a10, axis_labels=None, mmr=True, megno=True):
    Ns = 3
    ps = sim.particles
    for q, [label, i1, i2] in enumerate(pairs):
        m1 = ps[i1].m
        m2 = ps[i2].m
        e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
        e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
        tseries[i,Ns*q+1] = np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
        tseries[i,Ns*q+2] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
        if mmr:
            j, k, tseries[i,Ns*q+3] = find_strongest_MMR(sim, i1, i2) 
        else:
            tseries[i,Ns*q+3] = 0.0

        if axis_labels is not None:
            axis_labels[Ns*q+1] = 'e+_' + label
            axis_labels[Ns*q+2] = 'e-_' + label
            axis_labels[Ns*q+3] = 'max_strength_mmr_' + label


    if axis_labels is not None:
        axis_labels[7] = 'megno'

    if megno:
        tseries[i,7] = sim.calculate_megno() # megno
    else:
        tseries[i,7] = 0.0

    orbits = sim.calculate_orbits()
    for j, k in enumerate(trio):
        o = orbits[k-1]
        tseries[i, 8+6*j] = o.a/a10
        tseries[i, 9+6*j] = o.e
        tseries[i, 10+6*j] = o.inc
        tseries[i, 11+6*j] = o.Omega
        tseries[i, 12+6*j] = o.pomega
        tseries[i, 13+6*j] = o.theta
        if axis_labels is not None:
            axis_labels[8+6*j] = 'a' + str(j+1)
            axis_labels[9+6*j] = 'e' + str(j+1)
            axis_labels[10+6*j] = 'i' + str(j+1)
            axis_labels[11+6*j] = 'Omega' + str(j+1)
            axis_labels[12+6*j] = 'pomega' + str(j+1)
            axis_labels[13+6*j] = 'theta' + str(j+1)

@profile
def get_extended_tseries(sim, args, mmr=True, megno=True):
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
   
    a10s = [sim.particles[trio[0]].a for trio in trios]
    minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])

    # want hyperbolic case to run so it raises exception
    times = np.linspace(0, Norbits*minP, Nout)
    triopairs, triotseries = [], []
    # axis_labels = ['']*26
    # axis_labels[0] = 'time'
    #7 are same as used for SPOCK (equivalent of old res_tseries), and following 18 are the 6 orbital elements for each of the 3 planets. 
    axis_labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'Omega1', 'pomega1', 'theta1', 'a2', 'e2', 'i2', 'Omega2', 'pomega2', 'theta2', 'a3', 'e3', 'i3', 'Omega3', 'pomega3', 'theta3']

    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs 
        triopairs.append(get_pairs(sim, trio))
        triotseries.append(np.zeros((Nout, 26))*np.nan)
  
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except (rebound.Collision, rebound.Escape):
            stable = False
            return triotseries, sim.t/minP

        for tseries in triotseries:
            tseries[i,0] = sim.t/minP  # time

        for tr, trio in enumerate(trios):
            pairs = triopairs[tr]
            tseries = triotseries[tr] 
            populate_extended_trio(sim, trio, pairs, tseries, i, a10s[tr], mmr=mmr, megno=megno)
            # if i == 0 and tr == 0:
                # populate_extended_trio(sim, trio, pairs, tseries, i, a10s[tr], axis_labels)
            # else:
                # populate_extended_trio(sim, trio, pairs, tseries, i, a10s[tr])
    
    # print(axis_labels)
    #triotseries = pd.DataFrame(data=triotseries, columns=axis_labels)
    stable = True
    return triotseries, stable

def populate_trio(sim, trio, pairs, tseries, i):
    Ns = 3
    ps = sim.particles
    for q, [label, i1, i2] in enumerate(pairs):
        m1 = ps[i1].m
        m2 = ps[i2].m
        e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
        e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
        tseries[i,Ns*q+1] = np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
        tseries[i,Ns*q+2] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
        j, k, tseries[i,Ns*q+3] = find_strongest_MMR(sim, i1, i2) 

    tseries[i,7] = sim.calculate_megno() # megno

def get_tseries(sim, args):
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    minP = np.min([p.P for p in sim.particles[1:sim.N_real]])

    # want hyperbolic case to run so it raises exception
    times = np.linspace(0, Norbits*np.abs(minP), Nout)
    
    triopairs, triotseries = [], []

    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs 
        triopairs.append(get_pairs(sim, trio))
        triotseries.append(np.zeros((Nout, 8))*np.nan)
  
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except rebound.Collision:
            stable = False
            return triotseries, stable

        for tseries in triotseries:
            tseries[i,0] = sim.t/minP  # time

        for tr, trio in enumerate(trios):
            pairs = triopairs[tr]
            tseries = triotseries[tr] 
            populate_trio(sim, trio, pairs, tseries, i)
    
    stable = True
    return triotseries, stable
    
def features(sim, args):
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    ps  = sim.particles
    triofeatures = []
    for tr, trio in enumerate(trios):
        features = OrderedDict()
        pairs = get_pairs(sim, trio)
        for i, [label, i1, i2] in enumerate(pairs):
            features['EMcross'+label] = (ps[i2].a-ps[i1].a)/ps[i1].a
            features['EMfracstd'+label] = np.nan
            features['EPstd'+label] = np.nan
            features['MMRstrength'+label] = np.nan

        features['MEGNO'] = np.nan
        features['MEGNOstd'] = np.nan
        triofeatures.append(features)
    
    triotseries, stable = get_tseries(sim, args)
    if stable == False:
        return triofeatures, stable

    for features, tseries in zip(triofeatures, triotseries):
        EMnear = tseries[:, 1]
        EPnear = tseries[:, 2]
        # cut out first value (init cond) to avoid cases
        # where user sets exactly b*n2 - a*n1 & strength is inf
        MMRstrengthnear = tseries[1:,3]
        EMfar = tseries[:, 4]
        EPfar = tseries[:, 5]
        MMRstrengthfar = tseries[1:,6]
        MEGNO = tseries[:, 7]

        if not np.isnan(MEGNO).any(): # no nans
            features['MEGNO'] = np.median(MEGNO[-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
            features['MEGNOstd'] = MEGNO[int(Nout/5):].std()
        features['MMRstrengthnear'] = np.median(MMRstrengthnear)
        features['MMRstrengthfar'] = np.median(MMRstrengthfar)
        features['EMfracstdnear'] = EMnear.std() / features['EMcrossnear']
        features['EMfracstdfar'] = EMfar.std() / features['EMcrossfar']
        features['EPstdnear'] = EPnear.std() 
        features['EPstdfar'] = EPfar.std() 
    
    return triofeatures, stable
