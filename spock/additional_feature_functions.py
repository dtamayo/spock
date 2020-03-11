import rebound
import numpy as np
import pandas as pd
from collections import OrderedDict
from feature_functions import error_check, get_pairs, init_sim, find_strongest_MMR
from AMD import AMD_subset, AMD_crit

def additional_populate_trio(sim, trio, pairs, jk, a10, tseries, i):
    Ns = 3
    ps = sim.particles
    for q, [label, i1, i2] in enumerate(pairs):
        e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), -ps[i1].e*np.sin(ps[i1].pomega)
        e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), -ps[i2].e*np.sin(ps[i2].pomega)
        try:
            # it's unlikely but possible for bodies to land nearly on top of  each other midtimestep and get a big kick that doesn't get caught by collisions  post timestep. All these cases are unstable, so flag them as above
                # average only affects a (Lambda) Z and I think  Zcom  don't depend on a. Zsep and Zstar slightly, but several sig figs in even when close at conjunction
            j,k = jk[label]
            avars = Andoyer.from_Simulation(sim, a10=a10[label], j=j, k=k, i1=i1, i2=i2, average=False)
            tseries[i,Ns*q+1] = avars.Z*np.sqrt(2) # EM = Z*sqrt(2)
            tseries[i,Ns*q+2] = avars.Zcom # no sqrt(2) factor
        except: # no nearby resonance, use EM and ecom
            tseries[i,Ns*q+1] = np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            tseries[i,Ns*q+2] = np.sqrt((ps[i1].m*e1x + ps[i2].m*e2x)**2 + (ps[i1].m*e1y + ps[i2].m*e2y)**2)/(ps[i1].m+ps[i2].m)
        try:
            j, k, tseries[i,Ns*q+3] = find_strongest_MMR(sim, i1, i2) # will fail if any osculating orbit is hyperbolic, flag as unstable
        except:
            return 0

    tseries[i,7] = AMD_subset(sim, trio)
    tseries[i,8] = sim.calculate_megno() # megno

    return 1

def additional_get_tseries(sim, Norbits, Nout, trios, triopairs, triojks, trioa10s, triotseries):
    P0 = sim.particles[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    
    for tr, trio in enumerate(trios): 
        triotseries.append(np.zeros((Nout, 9)))
   
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except:
            return 0
    
        for tseries in triotseries:
            tseries[i,0] = sim.t/P0  # time

        for tr, trio in enumerate(trios):
            pairs = triopairs[tr]
            jk = triojks[tr]
            a10 = trioa10s[tr]
            tseries = triotseries[tr] 
            stable = additional_populate_trio(sim, trio, pairs, jk, a10, tseries, i)
            if not stable:
                return 0
    return 1
    
def additional_features(sim, args): # final cut down list
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    ps  = sim.particles

    triofeatures = []
    for tr, trio in enumerate(trios):
        error_check(sim, Norbits, Nout, trio)
        features = OrderedDict()
        pairs = get_pairs(sim, trio)
        for i, [label, i1, i2] in enumerate(pairs):
            features['EMfracstd'+label] = np.nan  # E = exact (celmech)
            features['EPstd'+label] = np.nan
            features['AMDcrit'+label] = np.nan
            features['AMDtriofrac'+label] = np.nan
            features['EMcross'+label] = np.nan
            features['MMRstrength'+label] = np.nan
            features['j'+label] = np.nan
            features['k'+label] = np.nan

        features['MEGNO'] = np.nan
        features['MEGNOstd'] = np.nan
        features['stableinshortintegration'] = 1

        for i, [label, i1, i2] in enumerate(pairs):
            RH = ps[i1].a*((ps[i1].m + ps[i2].m)/ps[0].m)**(1./3.)
            features['beta'+label] = (ps[i2].a-ps[i1].a)/RH
            features["AMDcrit"+label] = AMD_crit(sim, i1, i2)
            features["EMcross"+label] = (ps[i2].a-ps[i1].a)/ps[i1].a       
            features["j"+label], features["k"+label], _ = find_strongest_MMR(sim, i1, i2)
        
        triofeatures.append(features)
    
    triopairs, triojks, trioa10s, triotseries = init_sim(sim, trios)
    stable = additional_get_tseries(sim, Norbits, Nout, trios, triopairs, triojks, trioa10s, triotseries)

    if not stable:
        for features in triofeatures:
            features['stableinshortintegration'] = 0
        return triofeatures, False

    for features, tseries in zip(triofeatures, triotseries):
        EMnear = tseries[:, 1]
        EPnear = tseries[:, 2]
        MMRstrengthnear = tseries[:,3]
        EMfar = tseries[:, 4]
        EPfar = tseries[:, 5]
        MMRstrengthfar = tseries[:,6]
        AMDtrio = tseries[:, 7]
        MEGNO = tseries[:, 8]

        features['MEGNO'] = np.median(MEGNO[-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
        features['MEGNOstd'] = MEGNO.std()
        features['AMDtriofracnear'] = np.median(AMDtrio) / features['AMDcritnear']
        features['AMDtriofracfar'] = np.median(AMDtrio) / features['AMDcritfar']
        features['MMRstrengthnear'] = np.median(MMRstrengthnear)
        features['MMRstrengthfar'] = np.median(MMRstrengthfar)
        features['EMfracstdnear'] = EMnear.std() / features['EMcrossnear']
        features['EMfracstdfar'] = EMfar.std() / features['EMcrossfar']
        features['EPstdnear'] = EPnear.std() 
        features['EPstdfar'] = EPfar.std() 
        
    return triofeatures, True
