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

def critical_relative_AMD(alpha,gamma):
    """Equation 29"""
    e0 = np.min((1,1/alpha-1))
    ec = brenth(F,0,e0,args=(alpha,gamma))
    e1c = np.sin(np.arctan(gamma*ec / np.sqrt(alpha*(1-ec*ec))))
    curlyC = gamma*np.sqrt(alpha) * (1-np.sqrt(1-ec*ec)) + (1 - np.sqrt(1-e1c*e1c))
    return curlyC

def compute_AMD(sim):
    pstar = sim.particles[0]
    Ltot = pstar.m * np.cross(pstar.xyz,pstar.vxyz)
    ps = sim.particles[1:]
    Lmbda=np.zeros(len(ps))
    G = np.zeros(len(ps))
    Lhat = np.zeros((len(ps),3))
    for k,p in enumerate(sim.particles[1:]):
        orb = p.calculate_orbit(primary=pstar)
        Lmbda[k] = p.m * np.sqrt(p.a)
        G[k] = Lmbda[k] * np.sqrt(1-p.e*p.e)
        hvec = np.cross(p.xyz,p.vxyz)
        Lhat[k] = hvec / np.linalg.norm(hvec)
        Ltot = Ltot + p.m * hvec
    cosi = np.array([Lh.dot(Ltot) for Lh in Lhat]) / np.linalg.norm(Ltot)
    return np.sum(Lmbda) - np.sum(G * cosi)

def AMD_stable_Q(sim):
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        if C>Ccrit:
            return False
    return True

def AMD_stability_coefficients(sim):
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    coeffs = np.zeros(len(ps)-1)
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        coeffs[i] = C / Ccrit
    return coeffs

def AMD_stability_coefficient(sim, i1, i2):
    AMD = compute_AMD(sim)
    ps = sim.particles
    pstar = ps[0]
    
    pIn = ps[i1]
    pOut = ps[i2]
    orbIn = pIn.calculate_orbit(pstar)
    orbOut = pOut.calculate_orbit(pstar)
    alpha = orbIn.a / orbOut.a
    gamma = pIn.m / pOut.m
    LmbdaOut = pOut.m * np.sqrt(orbOut.a)
    Ccrit = critical_relative_AMD(alpha,gamma)
    C = AMD / LmbdaOut
    return C / Ccrit

### end AMD functions

def fillnanv6(features, pairs):
    features['tlyap'] = np.nan
    features['megno'] = np.nan

    for i, [label, i1, i2] in enumerate(pairs):
        features['EMmed'+label] = np.nan
        features['EMmax'+label] = np.nan
        features['EMstd'+label] = np.nan
        features['EMslope'+label] = np.nan
        features['EMrollingstd'+label] = np.nan
        features['EPmed'+label] = np.nan
        features['EPmax'+label] = np.nan
        features['EPstd'+label] = np.nan
        features['EPslope'+label] = np.nan
        features['EProllingstd'+label] = np.nan

def getpairsv5(sim):
    N = sim.N - sim.N_var
    Npairs = int((N-1)*(N-2)/2)
    EMcross = np.zeros(Npairs)
    ps = sim.particles
    #print('pairindex, i1, i2, j, k, strength')
    for i, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
        i1 = int(i1); i2 = int(i2)
        EMcross[i] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        
    if EMcross[0] < EMcross[2]: # 0 = '1-2', 2='2-3'
        pairs = [['near', 1,2], ['far', 2, 3], ['outer', 1, 3]]
    else:
        pairs = [['near', 2, 3], ['far', 1, 2], ['outer', 1, 3]]

    return pairs

def findresv3(sim, i1, i2):
    maxorder = 2
    ps = Poincare.from_Simulation(sim=sim).particles # get averaged mean motions
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[i1].M
    m2 = ps[i2].m/ps[i2].M

    Pratio = n2/n1
    if np.isnan(Pratio): # probably due to close encounter where averaging step doesn't converge
        return np.nan, np.nan, np.nan

    delta = 0.03
    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.999) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=2)

    Z = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    Zcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, i1, i2, maxstrength = -1, -1, -1, -1, -1
    for a, b in res:
        s = np.abs(np.sqrt(m1+m2)*(Z/Zcross)**((b-a)/2.)/((b*n2 - a*n1)/n1))
        #print('{0}:{1}'.format(b, a), (b*n2 - a*n1), s)
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s

    return j, k, maxstrength


def resparamsv5(sim):
    features = OrderedDict()
    ps = sim.particles
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    pairs = getpairsv5(sim)
    for i, [label, i1, i2] in enumerate(pairs):
        # recalculate with new ordering
        RH = ps[i1].a*((ps[i1].m + ps[i2].m)/ps[0].m)**(1./3.)
        features['beta'+label] = (ps[i2].a-ps[i1].a)/RH
        features['EMcross'+label] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        features['j'+label], features['k'+label], features['strength'+label] = findresv3(sim, i1, i2) # returns -1s if no res found
        features["C_AMD"+label] = AMD_stability_coefficient(sim, i1, i2)
        features['reshalfwidth'+label] = np.nan # so we always populate in case there's no  separatrix
        if features['strength'+label] > 0:
            pvars = Poincare.from_Simulation(sim)
            avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
            Zsepinner = avars.Zsep_inner # always positive, location at (-Zsepinner, 0)
            Zsepouter = avars.Zsep_outer
            Zstar = avars.Zstar
            features['reshalfwidth'+label] = min(Zsepouter-Zstar, Zstar-Zsepinner)
    
    sortedstrengths = np.array([features['strength'+label] for label in ['near', 'far', 'outer']])
    sortedstrengths.sort() # ascending
    
    if sortedstrengths[-1] > 0 and sortedstrengths[-2] > 0: # if two strongeest resonances are nonzereo
        features['secondres'] = sortedstrengths[-2]/sortedstrengths[-1] # ratio of second largest strength to largest
    else:
        features['secondres'] = -1
            
    return pd.Series(features, index=list(features.keys())) 


def ressummaryfeaturesxgbv6(sim): # don't use features that require transform to res variables in tseries
    Norbits = 10000
    Nout = 1000
    if sim.integrator != "whfast":
        sim.integrator = "whfast"
        sim.dt = 2*np.sqrt(3)/100.
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################
    ps = sim.particles
    sim.init_megno()
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    features = resparamsv5(sim)
    pairs = getpairsv5(sim)
    
    P0 = ps[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    Z, phi = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zcom, phiZcom = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zstar = np.zeros((Npairs,Nout))
    eminus, eplus = np.zeros((Npairs,Nout)), np.zeros((Npairs, Nout))
    
    features['unstableinNorbits'] = False
    AMD0 = 0
    for p in ps[1:sim.N-sim.N_var]:
        AMD0 += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))

    AMDerr = np.zeros(Nout)
    for i,t in enumerate(times):
        try:
            sim.integrate(t*P0, exact_finish_time=0)
        except:
            features['unstableinNorbits'] = True
            break
        AMD = 0
        for p in ps[1:sim.N-sim.N_var]:
            AMD += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))
        AMDerr[i] = np.abs((AMD-AMD0)/AMD0)
        
        for j, [label, i1, i2] in enumerate(pairs):
            eminus[j, i] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2) / features['EMcross'+label]
            eplus[j, i] = np.sqrt((ps[i1].m*ps[i1].e*np.cos(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].m*ps[i1].e*np.sin(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.sin(ps[i2].pomega))**2)/(ps[i1].m+ps[i2].m)
            
    fillnanv6(features, pairs)
    
    Nnonzero = int((eminus[0,:] > 0).sum())
    times = times[:Nnonzero]
    AMDerr = AMDerr[:Nnonzero]
    
    eminus = eminus[:,:Nnonzero]
    eplus = eplus[:,:Nnonzero]
    
    # Features with or without resonances:
    tlyap = 1./sim.calculate_lyapunov()/P0
    if tlyap < 0 or tlyap > Norbits:
        tlyap = Norbits
    features['tlyap'] = tlyap
    features['megno'] = sim.calculate_megno()
    features['AMDerr'] = AMDerr.max()
    
    for i, [label, i1, i2] in enumerate(pairs):
        EM = eminus[i,:]
        EP = eplus[i,:]
        features['EMmed'+label] = np.median(EM)
        features['EMmax'+label] = EM.max()
        features['EMstd'+label] = EM.std()
        features['EPmed'+label] = np.median(EP)
        features['EPmax'+label] = EP.max()
        features['EPstd'+label] = EP.std()
        
        last = np.median(EM[-int(Nout/20):])
        features['EMslope'+label] = (last - EM.min())/EM.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
        
        rollstd = pd.Series(EM).rolling(window=10).std()
        features['EMrollingstd'+label] = rollstd[10:].median()/features['EMmed'+label]
        
        last = np.median(EP[-int(Nout/20):])
        features['EPslope'+label] = (last - EP.min())/EP.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
            
        rollstd = pd.Series(EP).rolling(window=10).std()
        features['EProllingstd'+label] = rollstd[10:].median()/features['EPmed'+label]
        
    return pd.Series(features, index=list(features.keys())) 
