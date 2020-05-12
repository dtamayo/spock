import rebound
import numpy as np
import itertools
from scipy.optimize import brenth
from collections import OrderedDict
from celmech import Andoyer
from celmech.resonances import resonant_period_ratios
from celmech.disturbing_function import get_fg_coeffs

fn,gn,beta1,beta2 = np.zeros((100,3)), np.zeros((100,3)), np.zeros((100,3)), np.zeros((100,3))
for k in range(1,3):
    for j in range(k+1,100):
        f, g = get_fg_coeffs(j,k)
        norm = np.sqrt(f**2 + g**2)
        fn[j,k] = f/norm
        gn[j,k] = g/norm
        rootalphainv = (j/(j-k))**(1/3) # sqrt(a20/a10s)
        beta1[j,k] = np.abs(f/g)*rootalphainv
        beta2[j,k] = (j-k)/j*rootalphainv

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis
def get_pairs(sim, indices):
    ps = sim.particles
    sortedindices = sorted(indices, key=lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = (ps[sortedindices[1]].a-ps[sortedindices[0]].a)/ps[sortedindices[0]].a
    EMcrossOuter = (ps[sortedindices[2]].a-ps[sortedindices[1]].a)/ps[sortedindices[1]].a

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]], ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]], ['far', sortedindices[0], sortedindices[1]]]

def find_strongest_MMR(sim, i1, i2):
    maxorder = 2
    ps = sim.particles
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[0].m
    m2 = ps[i2].m/ps[0].m

    Pratio = n2/n1

    delta = 0.03
    if Pratio < 0 or Pratio > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan

    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=2)

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    EMcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, maxstrength = np.nan, np.nan, 0 
    for a, b in res:
        s = np.abs(np.sqrt(m1+m2)*(EM/EMcross)**((b-a)/2.)/((b*n2 - a*n1)/n1))
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s
    if maxstrength == 0:
        maxstrength = np.nan

    return j, k, maxstrength

def init_lists(sim, trios):
    triopairs, triojks, trioa10s, triotseries = [], [], [], []
    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs that have their own j,k,strength
        triopairs.append(get_pairs(sim, trio))
        triojks.append({})
        trioa10s.append({})
        for [label, i1, i2] in triopairs[tr]:
            j, k, _ = find_strongest_MMR(sim, i1, i2) 
            if np.isnan(j) == False:
                triojks[tr][label] = (j,k)
                trioa10s[tr][label] = sim.particles[i1].a

    return triopairs, triojks, trioa10s, triotseries 

def populate_trio(sim, trio, pairs, jk, a10, tseries, i):
    Ns = 5
    ps = sim.particles
    for q, [label, i1, i2] in enumerate(pairs):
        m1 = ps[i1].m
        m2 = ps[i2].m
        e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
        e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
        tseries[i,Ns*q+1] = np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
        tseries[i,Ns*q+2] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
        try:
            # it's unlikely but possible for bodies to land nearly on top of  each other midtimestep and get a big kick that doesn't get caught by collisions  post timestep. All these cases are unstable, so flag them as above
                # average only affects a (Lambda) Z and I think  Zcom  don't depend on a. Zsep and Zstar slightly, but several sig figs in even when close at conjunction
            j,k = jk[label]
            tseries[i,Ns*q+3] = np.sqrt((fn[j,k]*e1x+gn[j,k]*e2x)**2 + (fn[j,k]*e1y+gn[j,k]*e2y)**2)*np.sqrt(2) # EM = Z*sqrt(2)
            tseries[i,Ns*q+4] = np.sqrt((m1*e1x + beta1[j,k]*m2*e2x)**2 + (m1*e1y + beta1[j,k]*m2*e2y)**2)/(m1+beta2[j,k]*m2) # no sqrt(2) factor
        except: # no nearby resonance, use EM and ecom
            tseries[i,Ns*q+3] = np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            tseries[i,Ns*q+4] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
        j, k, tseries[i,Ns*q+5] = find_strongest_MMR(sim, i1, i2) 

    tseries[i,11] = sim.calculate_megno() # megno

def get_tseries(sim, args):
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]

    P0 = sim.particles[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    
    triopairs, triojks, trioa10s, triotseries = init_lists(sim, trios)
    
    for tr, trio in enumerate(trios): 
        triotseries.append(np.zeros((Nout, 12))*np.nan)
   
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except:
            pass

        if sim._status == 5: # checking this way works for both new rebound and old version used for random dataset
            return triotseries

        for tseries in triotseries:
            tseries[i,0] = sim.t/P0  # time

        for tr, trio in enumerate(trios):
            pairs = triopairs[tr]
            jk = triojks[tr]
            a10 = trioa10s[tr]
            tseries = triotseries[tr] 
            populate_trio(sim, trio, pairs, jk, a10, tseries, i)

    return triotseries 
    
def features(sim, args): # final cut down list
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    ps  = sim.particles
    triofeatures = []
    for tr, trio in enumerate(trios):
        #error_check(sim, Norbits, Nout, trio)
        features = OrderedDict()
        pairs = get_pairs(sim, trio)
        for i, [label, i1, i2] in enumerate(pairs):
            features['EMcross'+label] = (ps[i2].a-ps[i1].a)/ps[i1].a
            features['EMfracstd'+label] = np.nan
            features['EPstd'+label] = np.nan
            features['EMfracstdapprox'+label] = np.nan
            features['EPstdapprox'+label] = np.nan
            features['MMRstrength'+label] = np.nan

        features['MEGNO'] = np.nan
        features['MEGNOstd'] = np.nan
        features['stable_in_short_integration'] = False
        triofeatures.append(features)#pd.Series(features, index=list(features.keys())))
    
    triotseries = get_tseries(sim, args)
    if sim._status == 5: # unstable
        return triofeatures

    for features, tseries in zip(triofeatures, triotseries):
        EMapproxnear = tseries[:, 1]
        EPapproxnear = tseries[:, 2]
        EMnear = tseries[:, 3]
        EPnear = tseries[:, 4]
        MMRstrengthnear = tseries[:,5]
        EMapproxfar = tseries[:, 6]
        EPapproxfar = tseries[:, 7]
        EMfar = tseries[:, 8]
        EPfar = tseries[:, 9]
        MMRstrengthfar = tseries[:,10]
        MEGNO = tseries[:, 11]
        
        features['stable_in_short_integration'] = True
        features['MEGNO'] = np.median(MEGNO[-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
        features['MEGNOstd'] = MEGNO[int(Nout/5):].std()
        features['MMRstrengthnear'] = np.median(MMRstrengthnear)
        features['MMRstrengthfar'] = np.median(MMRstrengthfar)
        features['EMfracstdapproxnear'] = EMapproxnear.std() / features['EMcrossnear']
        features['EMfracstdapproxfar'] = EMapproxfar.std() / features['EMcrossfar']
        features['EPstdapproxnear'] = EPapproxnear.std() 
        features['EPstdapproxfar'] = EPapproxfar.std() 
        features['EMfracstdnear'] = EMnear.std() / features['EMcrossnear']
        features['EMfracstdfar'] = EMfar.std() / features['EMcrossfar']
        features['EPstdnear'] = EPnear.std() 
        features['EPstdfar'] = EPfar.std() 
        
    return triofeatures
