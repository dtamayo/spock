import rebound
import numpy as np
import pandas as pd
from collections import OrderedDict
from celmech.poincare import Poincare, PoincareHamiltonian
from celmech import Andoyer, AndoyerHamiltonian
from celmech.resonances import resonant_period_ratios, resonance_intersections_list, resonance_pratio_span
from celmech.transformations import masses_to_jacobi
from celmech.andoyer import get_num_fixed_points
import itertools

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0

from scipy.optimize import brenth
def F(e,alpha,gamma):
    """Equation 35 of Laskar & Petit (2017)"""
    denom = np.sqrt(alpha*(1-e*e)+gamma*gamma*e*e)
    return alpha*e -1 + alpha + gamma*e / denom

### start AMD functions

def spock_AMD_crit(sim, i1, i2): # assumes i1.a < i2.a
    # returns AMD_crit dimensionalized to Lambdaout
    ps = sim.particles
    if ps[i1].m == 0. or ps[i2].m == 0:
        return 0 # if one particle is massless, any amount of AMD can take it to e=1, so AMDcrit = 0 (always AMD unstable)

    mu = sim.G*ps[0].m
    alpha = ps[i1].a / ps[i2].a
    gamma = ps[i1].m / ps[i2].m
    LambdaPrime = ps[i2].m * np.sqrt(mu*ps[i2].a)
    curlyC = spock_relative_AMD_crit(alpha,gamma)
    AMD_crit = curlyC * LambdaPrime # Eq 29 AMD_crit = C = curlyC*Lambda'

    return AMD_crit

def spock_relative_AMD_crit(alpha,gamma):
    """Equation 29"""
    e0 = np.min((1,1/alpha-1))
    try:
        ec = brenth(F,0,e0,args=(alpha,gamma))
    except: # can fail to converge
        print('brenth failed: e0  = {0}, alpha = {1}, gamma= {2}'.format(e0, alpha, gamma))
        return np.nan
    e1c = np.sin(np.arctan(gamma*ec / np.sqrt(alpha*(1-ec*ec))))
    curlyC = gamma*np.sqrt(alpha) * (1-np.sqrt(1-ec*ec)) + (1 - np.sqrt(1-e1c*e1c))
    return curlyC

def spock_AMDsubset(sim, indices): 
    ps = sim.particles
    sim2 = rebound.Simulation()
    sim2.G = sim.G
    sim2.add(ps[0].copy())
    for i in indices:
        sim2.add(ps[i].copy())
    ps = sim2.particles
    Lx, Ly, Lz = sim2.calculate_angular_momentum()
    L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lcirc = 0
    Mint = ps[0].m
    for p in ps[1:sim.N_real]: # the exact choice of which a and masses to use doesn't matter for closely packed systems (only hierarchical)
        mred = p.m*Mint/(p.m+Mint)
        Lcirc += mred * np.sqrt(sim.G*(p.m + Mint)*p.a)
        Mint += p.m
    return Lcirc - L

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis
def spock_3p_pairs(sim, indices):
    ps = sim.particles
    sortedindices = sorted(indices, key=lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = (ps[sortedindices[1]].a-ps[sortedindices[0]].a)/ps[sortedindices[0]].a
    EMcrossOuter = (ps[sortedindices[2]].a-ps[sortedindices[1]].a)/ps[sortedindices[1]].a

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]], ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]], ['far', sortedindices[0], sortedindices[1]]]

def spock_find_strongest_MMR(sim, i1, i2):
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
    maxperiodratio = min(Pratio+delta, 0.999) # too many resonances close to 1
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

def spock_error_check(sim, Norbits, Nout, trio):
    i1, i2, i3 = trio
    if not isinstance(i1, int) or not isinstance(i2, int) or not isinstance(i3, int):
        raise AttributeError("SPOCK  Error: Particle indices passed to spock_features were not integers")
    ps = sim.particles
    if ps[0].m < 0 or ps[1].m < 0 or ps[2].m < 0 or ps[3].m < 0: 
        raise AttributeError("SPOCK Error: Particles in sim passed to spock_features had negative masses")

    if ps[1].r == 0 or ps[2].r == 0 or ps[3].r == 0.:
        for  p in ps[1:]:
            rH = p.a*(p.m/3./ps[0].m)**(1./3.)
            p.r = rH
    return

def spock_init_sim(sim, trios, Nout):
    # AMD calculation is easiest in canonical heliocentric coordiantes, so velocities need to be in barycentric frame
    # Don't want to move_to_com() unless we have to so that we get same chaotic trajectory as user passes
    com = sim.calculate_com()
    if com.x**2 + com.y**2 + com.z**2 + com.vx**2 + com.vy**2 + com.vz**2 > 1.e-16:
        sim.move_to_com()
    
    ###############################
    try:
        sim.collision = 'line' # use line if using newer version of REBOUND
    except:
        sim.collision = 'direct'# fall back for older versions
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################
    ps = sim.particles
    try:
        sim.init_megno(seed=0)
    except:
        sim.init_megno()
   
    if sim.integrator != "whfast":
        sim.integrator = "whfast"
        sim.dt = 2*np.sqrt(3)/100.*sim.particles[1].P

    triopairs, triojks, trioa10s, triotseries = [], [], [], []
    for tr, trio in enumerate(trios): # For each trio there are two adjacent pairs that have their own j,k,strength
        triopairs.append(spock_3p_pairs(sim, trio))
        triojks.append({})
        trioa10s.append({})
        triotseries.append(np.zeros((Nout, 9)))
        for [label, i1, i2] in triopairs[tr]:
            j, k, _ = spock_find_strongest_MMR(sim, i1, i2) 
            if np.isnan(j) == False:
                triojks[tr][label] = (j,k)
                trioa10s[tr][label] = ps[i1].a

    return triopairs, triojks, trioa10s, triotseries 

def spock_populate_trio(sim, trio, pairs, jk, a10, tseries, i):
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
            j, k, tseries[i,Ns*q+3] = spock_find_strongest_MMR(sim, i1, i2) # will fail if any osculating orbit is hyperbolic, flag as unstable
        except:
            return 0

    tseries[i,7] = spock_AMDsubset(sim, trio)
    tseries[i,8] = sim.calculate_megno() # megno

    return 1

def spock_3p_tseries(sim, Norbits, Nout, trios, triopairs, triojks, trioa10s, triotseries):
    P0 = sim.particles[1].P
    times = np.linspace(0, Norbits*P0, Nout)
   
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
            stable = spock_populate_trio(sim, trio, pairs, jk, a10, tseries, i)
            if not stable:
                return 0
    return 1
    
def spock_features(sim, args): # final cut down list
    Norbits = args[0]
    Nout = args[1]
    trios = args[2]
    
    ps  = sim.particles

    triofeatures = []
    for tr, trio in enumerate(trios):
        spock_error_check(sim, Norbits, Nout, trio)
        features = OrderedDict()
        pairs = spock_3p_pairs(sim, trio)
        for i, [label, i1, i2] in enumerate(pairs):
            features['EMfracstd'+label] = np.nan  # E = exact (celmech)
            features['EPstd'+label] = np.nan
            features['AMDcrit'+label] = np.nan
            features['AMDtriofrac'+label] = np.nan
            features['EMcross'+label] = np.nan
            features['MMRstrength'+label] = np.nan
            features['MMRstrengthfilt'+label] = np.nan
            features['j'+label] = np.nan
            features['k'+label] = np.nan

        features['MEGNO'] = np.nan
        features['MEGNOstd'] = np.nan
        features['stableinshortintegration'] = 1

        for i, [label, i1, i2] in enumerate(pairs):
            features["AMDcrit"+label] = spock_AMD_crit(sim, i1, i2)
            features["EMcross"+label] = (ps[i2].a-ps[i1].a)/ps[i1].a       
            features["j"+label], features["k"+label], _ = spock_find_strongest_MMR(sim, i1, i2)
        
        triofeatures.append(pd.Series(features, index=list(features.keys())))
    
    triopairs, triojks, trioa10s, triotseries = spock_init_sim(sim, trios, Nout)
    stable = spock_3p_tseries(sim, Norbits, Nout, trios, triopairs, triojks, trioa10s, triotseries)

    if not stable:
        for features in triofeatures:
            features['stableinshortintegration'] = 0
        return triofeatures

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
        
    return triofeatures
