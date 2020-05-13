import rebound
import numpy as np
from scipy.optimize import brenth

def F(e,alpha,gamma):
    """Equation 35 of Laskar & Petit (2017)"""
    denom = np.sqrt(alpha*(1-e*e)+gamma*gamma*e*e)
    return alpha*e -1 + alpha + gamma*e / denom

### start AMD functions

def AMD_crit(sim2, i1, i2): # assumes i1.a < i2.a
    try:
        sim = sim2.copy()
        sim.move_to_com() # AMD calculations easiest in canonical heliocentric coordinates, but don't want to change original sim so copy
    except: # old version of REBOUND doesn't have a copy function (for random dataset), but all those were already moved to com so OK
        sim = sim2
    # returns AMD_crit dimensionalized to Lambdaout
    ps = sim.particles
    if ps[i1].m == 0. or ps[i2].m == 0:
        return 0 # if one particle is massless, any amount of AMD can take it to e=1, so AMDcrit = 0 (always AMD unstable)

    mu = sim.G*ps[0].m
    alpha = ps[i1].a / ps[i2].a
    gamma = ps[i1].m / ps[i2].m
    LambdaPrime = ps[i2].m * np.sqrt(mu*ps[i2].a)
    curlyC = relative_AMD_crit(alpha,gamma)
    AMD_crit = curlyC * LambdaPrime # Eq 29 AMD_crit = C = curlyC*Lambda'

    return AMD_crit

def relative_AMD_crit(alpha,gamma):
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

def AMD(sim2):
    try:
        sim = sim2.copy()
        sim.move_to_com() # AMD calculations easiest in canonical heliocentric coordinates, but don't want to change original sim so copy
    except: # old version of REBOUND doesn't have a copy function (for random dataset), but all those were already moved to com so OK
        sim = sim2
    ps = sim.particles
    Lx, Ly, Lz = sim.calculate_angular_momentum()
    L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lcirc = 0
    Mint = ps[0].m
    for p in ps[1:sim.N_real]: # the exact choice of which a and masses to use doesn't matter for closely packed systems (only hierarchical)
        mred = p.m*Mint/(p.m+Mint)
        Lcirc += mred * np.sqrt(sim.G*(p.m + Mint)*p.a)
        Mint += p.m
    return Lcirc - L
