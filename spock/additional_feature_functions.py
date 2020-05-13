import rebound
import numpy as np
import pandas as pd
from collections import OrderedDict
from spock.feature_functions import get_pairs, find_strongest_MMR, populate_trio
from spock.AMD_functions import AMD, AMD_crit

def additional_get_tseries(sim, args):
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]

    P0 = sim.particles[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    
    triopairs, triotseries = [], []
    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs 
        triopairs.append(get_pairs(sim, trio))
        triotseries.append(np.zeros((Nout, 9))*np.nan)
   
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except:
            pass

        if sim._status == 5: # checking this way works for both new rebound and old version used for random dataset
            stable = False
            return triotseries, stable

        for tseries in triotseries:
            tseries[i,0] = sim.t/P0  # time

        for tr, trio in enumerate(trios):
            pairs = triopairs[tr]
            tseries = triotseries[tr] 
            populate_trio(sim, trio, pairs, tseries, i)
            tseries[i,8] = AMD(sim)

    stable = True
    return triotseries, stable
    
def additional_features(sim, args): # final cut down list
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    ps  = sim.particles
    triofeatures = []
    for tr, trio in enumerate(trios):
        features = OrderedDict()
        pairs = get_pairs(sim, trio)
        for i, [label, i1, i2] in enumerate(pairs):
            features['EMfracstd'+label] = np.nan
            features['EPstd'+label] = np.nan
            features['AMDfrac'+label] = np.nan
            features['MMRstrength'+label] = np.nan
            
            RH = ps[i1].a*((ps[i1].m + ps[i2].m)/ps[0].m)**(1./3.)
            features['beta'+label] = (ps[i2].a-ps[i1].a)/RH
            features["AMDcrit"+label] = AMD_crit(sim, i1, i2)
            features["EMcross"+label] = (ps[i2].a-ps[i1].a)/ps[i1].a       
            features["j"+label], features["k"+label], _ = find_strongest_MMR(sim, i1, i2) 

        features['MEGNO'] = np.nan
        features['MEGNOstd'] = np.nan
        features['stable_in_short_integration'] = False

        triofeatures.append(features)
    
    features["e1Z07"] = ps[1].e * (ps[2].a+ps[1].a) / (ps[2].a-ps[1].a)
    features["e2Z07"] = ps[2].e * (ps[3].a+ps[2].a) / (ps[3].a-ps[2].a)
    features["e3Z07"] = ps[3].e * (ps[3].a+ps[2].a) / (ps[3].a-ps[2].a)
    features["eavgZ07inner"] = np.mean([features['e1Z07'], features['e2Z07']])
    features["eavgZ07outer"] = np.mean([features['e2Z07'], features['e3Z07']])
    features["eavgZ07"] = np.mean([features["e1Z07"], features["e2Z07"], features["e3Z07"]])
    features["muavgZ07inner"] = (ps[1].m+ps[2].m)/ps[0].m/2 # mean of the mass ratios
    features["muavgZ07outer"] = (ps[2].m+ps[3].m)/ps[0].m/2 # mean of the mass ratios
    features["muavgZ07"] = np.mean([ps[1].m, ps[2].m, ps[3].m])
    features["kZ07inner"] = (ps[2].a-ps[1].a) * 2. / (ps[2].a + ps[1].a) / (2.*features["muavgZ07inner"]/3.)**(1./3.)
    features["kZ07outer"] = (ps[3].a-ps[2].a) * 2. / (ps[3].a + ps[2].a) / (2.*features["muavgZ07outer"]/3.)**(1./3.)
    features["kZ07avg"] = np.mean([features['kZ07inner'], features['kZ07outer']])
    features["AZ07inner"] = -2. + features["eavgZ07inner"] - 0.27*np.log10(features["muavgZ07inner"]) # Zhou 2007 Eq 4
    features["AZ07outer"] = -2. + features["eavgZ07outer"] - 0.27*np.log10(features["muavgZ07outer"]) # Zhou 2007 Eq 4
    features["AZ07avg"] = -2. + features["eavgZ07"] - 0.27*np.log10(features["muavgZ07"]) # Zhou 2007 Eq 4
    features["BZ07inner"] = 18.7 + 1.1*np.log10(features["muavgZ07inner"]) - (16.8 + 1.2*np.log10(features["muavgZ07inner"]))*features['eavgZ07inner'] # Zhou 2007 Eq 4
    features["BZ07outer"] = 18.7 + 1.1*np.log10(features["muavgZ07outer"]) - (16.8 + 1.2*np.log10(features["muavgZ07outer"]))*features['eavgZ07outer'] # Zhou 2007 Eq 4
    features["BZ07avg"] = 18.7 + 1.1*np.log10(features["muavgZ07"]) - (16.8 + 1.2*np.log10(features["muavgZ07"]))*features['eavgZ07'] # Zhou 2007 Eq 4
    features["Z07log_instability_time_inner"] = features["AZ07inner"] + features['BZ07inner']*np.log10(features['kZ07inner']/2.3)
    features["Z07log_instability_time_outer"] = features["AZ07outer"] + features['BZ07outer']*np.log10(features['kZ07outer']/2.3)
    features["Z07log_instability_time_avg"] = features["AZ07avg"] + features['BZ07avg']*np.log10(features['kZ07avg']/2.3)
    features["Z07Stable_avg"] = features["Z07log_instability_time_avg"] > 9
    features["Z07Stable_worstpair"] = min(features["Z07log_instability_time_inner"], features["Z07log_instability_time_outer"]) > 9

    features["deltaQ11inner"] = (ps[2].a-ps[1].a)/ps[2].a
    features["deltaQ11outer"] = (ps[3].a-ps[2].a)/ps[3].a
    features["deltaQ11avg"] = np.mean([features['deltaQ11inner'], features['deltaQ11outer']])
    features["Q11log_instability_time_inner"] = np.log10(features["deltaQ11inner"]**8 / np.abs(np.log(features["deltaQ11inner"]))**3 / features["muavgZ07inner"]**3 / 8.) # Qullen 2011 Eq 68
    features["Q11log_instability_time_outer"] = np.log10(features["deltaQ11outer"]**8 / np.abs(np.log(features["deltaQ11outer"]))**3 / features["muavgZ07outer"]**3 / 8.) # Qullen 2011 Eq 68
    features["Q11log_instability_time_avg"] = np.log10(features["deltaQ11avg"]**8 / np.abs(np.log(features["deltaQ11avg"]))**3 / features["muavgZ07"]**3 / 8.) # Qullen 2011 Eq 68
    features["Q11Stable_avg"] = features['Q11log_instability_time_avg'] > 9
    features["Q11Stable_worstpair"] = min(features['Q11log_instability_time_inner'], features["Q11log_instability_time_outer"]) > 9

    triotseries, stable = additional_get_tseries(sim, args)
    if not stable:
        return triofeatures, stable

    for features, tseries in zip(triofeatures, triotseries):
        EMnear = tseries[:, 1]
        EPnear = tseries[:, 2]
        MMRstrengthnear = tseries[:,3]
        EMfar = tseries[:, 4]
        EPfar = tseries[:, 5]
        MMRstrengthfar = tseries[:,6]
        MEGNO = tseries[:, 7]
        AMD = tseries[:, 8]

        features['MEGNO'] = np.median(MEGNO[-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
        features['MEGNOstd'] = MEGNO[int(Nout/5):].std()
        features['AMDfracnear'] = np.median(AMD) / features['AMDcritnear']
        features['AMDfracfar'] = np.median(AMD) / features['AMDcritfar']
        features['MMRstrengthnear'] = np.median(MMRstrengthnear)
        features['MMRstrengthfar'] = np.median(MMRstrengthfar)
        features['EMfracstdnear'] = EMnear.std() / features['EMcrossnear']
        features['EMfracstdfar'] = EMfar.std() / features['EMcrossfar']
        features['EPstdnear'] = EPnear.std() 
        features['EPstdfar'] = EPfar.std() 
        
    return triofeatures, stable
