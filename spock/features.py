from collections import OrderedDict
import numpy as np
import math
import rebound
from celmech.resonances import resonance_pratio_span
from celmech.resonances import resonance_jk_list

class Trio:
    def __init__(self, trio_indices, sim, Nout):
        '''initializes new set of features.

            trio_indices: numerical indices for the planets in the trio
            note: runningList stores all the time series over the short integration
                  second dict is for final features
                  all values get initialized to nan and get populated afterward
        '''
        # We keep track of the this trio and the adjacent pairs
        self.trio = trio_indices
        self.pairs = get_pairs(sim, trio_indices)

        # initialize running list which keeps track of data during simulation
        self.runningList = OrderedDict()

        # add keys here for time series that belong to the trio as a whole
        self.runningList['time'] = [np.nan] * Nout
        self.runningList['MEGNO'] = [np.nan] * Nout

        # Make the ordered dictionary to keep track of theta values
        self.theta = OrderedDict()

        # add keys here for time series that belong to each pair in the trio
        for each in ['Max','Min']:
            self.runningList['EM' + each] = [np.nan] * Nout
            self.runningList['EP' + each] = [np.nan] * Nout
            self.runningList['MMRstrength' + each] = [np.nan] * Nout
            self.runningList['pRat' + each] = [np.nan] * Nout 
            self.runningList['mu1' + each] = [np.nan] * Nout
            self.runningList['mu2' + each] = [np.nan] * Nout
            self.theta['order' + each] = []
            self.theta['vector' + each] = []
            self.theta['pRatio' + each] = []
            self.theta['relvector' + each] = []


        # dict of features to calculate

        self.features = OrderedDict()

        # add keys here for features that belong to each pair in the trio
        for each in ['Max', 'Min']:
            self.features['EMcross' + each] = np.nan
            self.features['EMfracstd' + each] = np.nan
            self.features['EPstd' + each] = np.nan
            self.features['MMRstrength' + each] = np.nan
            self.features['2BRFillFac' + each] = np.nan
            self.features['conjunctionMag' + each] = np.nan
            self.features['relConjunctionMag' + each] = np.nan


        # add keys here for features that belong to trio as a whole
        self.features['MEGNO'] = np.nan
        self.features['MEGNOstd'] = np.nan

    def fill_starting_features(self, sim):
        '''Fill the features that only depend on initial conditions
           sim is passed with the initial state before the short integration is run'''
        ps = sim.particles
        for [label, i1, i2] in self.pairs:
            # calculate crossing eccentricity
            self.features['EMcross' + label] = (ps[i2].a - ps[i1].a) / ps[i1].a
            pRat = getIntPrat(sim, i1, i2)
            self.theta['pRatio' + label] = pRat
            self.theta['order' + label] = pRat[1] - pRat[0]
            self.theta['vector' + label] = np.zeros(pRat[1] - pRat[0] + 1, dtype = complex)
            self.theta['relvector' + label] = 0.0j
        
        # calculate secular timescale and adds feature
        self.features['Tsec']= get_min_secT_trio(sim, self.trio)

    def fill_tseries_entry(self, sim, minP,i):
        '''Populates the runningList data dictionary for one time step.
           Fill in all entries that are tracked through the short integration

           minP: minimum orbital period among planets (needs passing since doesn't necessarily belong to trio)

           Note: must specify how each feature is calculated and added
        '''
        ps = sim.particles

        for q, [label, i1, i2] in enumerate(self.pairs):
            m1 = ps[i1].m
            m2 = ps[i2].m
            #calculate eccentricity vector
            e1x, e1y = ps[i1].e * np.cos(ps[i1].pomega), ps[i1].e * np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e * np.cos(ps[i2].pomega), ps[i2].e * np.sin(ps[i2].pomega)
            erel = ps[i2].e*np.exp(ps[i2].pomega*1j)-ps[i1].e*np.exp(ps[i1].pomega*1j)
            self.runningList['time'][i]= sim.t/minP
            #crossing eccentricity
            self.runningList['EM'+label][i] = np.sqrt((e2x - e1x)**2 + (e2y - e1y)**2)
            #mass weighted crossing eccentricity
            self.runningList['EP'+label][i] = np.sqrt((m1 * e1x + m2 * e2x)**2 +
                                                      (m1 * e1y + m2 * e2y)**2) / (m1+m2)
            #calculate the strength of MMRs
            MMRs = find_strongest_MMR_width(sim, i1, i2)
            self.runningList['MMRstrength' + label][i] = MMRs[2]

            # save mass ratios and integer period ratios
            self.runningList['mu1' + label][i] = m1 / ps[0].m
            self.runningList['mu2' + label][i] = m2 / ps[0].m
            self.runningList['pRat' + label][i] = ps[i1].P / ps[i2].P

            # calculates the conjunction angle based on each possible formula
            order = self.theta['order' + label]
            for o in range(order + 1):
                self.theta['vector' + label][o] += calcThetaVec(
                    ps[i1].l, 
                    ps[i1].pomega,
                    o,
                    ps[i2].l,
                    ps[i2].pomega,
                    order - o,
                    self.theta['pRatio' + label]
                    )
            self.theta['relvector' + label] += calcThetaRelVec(
                ps[i1].l, ps[i2].l, 
                self.theta['pRatio' + label],
                np.angle(erel)
                )


        # check rebound version, if old use .calculate_megno, otherwise use .megno, old is just version less then 4
        if float(rebound.__version__[0]) < 4:
            self.runningList['MEGNO'][i] = sim.calculate_megno()
        else:
            self.runningList['MEGNO'][i] = sim.megno()

    def fill_final_features(self, sim):
        '''fills the final set of features that are returned to the ML model.
           sim is passed with the final state after the short integration is run

            Each feature is filled depending on some combination of runningList features and initial condition features
        '''
        Nout = len(self.runningList['MEGNO'])

        if not np.isnan(self.runningList['MEGNO']).any(): # no nans
            # smooth last 10% to remove oscillations around 2
            self.features['MEGNO'] = np.median(
                self.runningList['MEGNO'][-(Nout // 10):]
            )

            self.features['MEGNOstd'] = np.std(
                self.runningList['MEGNO'][(Nout // 5):]
            )

        for label in ['Max', 'Min']:
            # cut out first value (init cond) to avoid cases
            # where user sets exactly b * n2 - a * n1 and strength is inf
            self.features['MMRstrength' + label] = np.median(
                self.runningList['MMRstrength' + label][1:]
            )
            self.features['EMfracstd' + label] = (
                np.std(self.runningList['EM' + label])
                / self.features['EMcross' + label])

            self.features['EPstd' + label] = \
                np.std(self.runningList['EP' + label])
            
            self.features['2BRFillFac' + label] = twoBRFillFac( 
                                                            np.nanmean(self.runningList['pRat' + label]),
                                                            np.nanmean(self.runningList['mu1' + label]),
                                                            np.nanmean(self.runningList['mu2' + label]),
                                                            np.nanmean(self.runningList['EM' + label])
                                                            )
            self.features['conjunctionMag' + label] = np.max(np.abs(self.theta['vector' + label])) / Nout
            self.features['relConjunctionMag' + label] = np.abs(self.theta['relvector' + label]) / Nout

def get_min_secT(sim):
    minList = []
    for trio_indices in [[j, j+1, j+2] for j in range(1, sim.N_real - 2)]:  # list of adjacent trios
        minList.append(get_min_secT_trio(sim, trio_indices))                # gets min secular time for that trio
    return min(minList)                                                     # returns min among all trios

def get_min_secT_trio(sim, trio):
    '''Calculates the secular time scale for a given trio in a simulation

        Arguments:
            sim: the simulation that contains the trio who's
                secular time scale you want
            trio: the trio who's secular timescale you want,
                Note: should be a list of body indexes

        Note: Calculated following Yang & Tamayo 2024
    '''
    ps = sim.particles
    p1, p2, p3 = ps[trio[0]], ps[trio[1]], ps[trio[2]]
    # determine the smallest period that a particle in the system has
    minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
    mStar = ps[0].m  # star should be the zero indexed body
    m1 = p1.m
    m2 = p2.m
    m3 = p3.m
    m_tot = m1 + m2 + m3
    mu1 = m1 / m_tot
    mu3 = m3 / m_tot
    alpha12 = p1.a / p2.a
    alpha23 = p2.a / p3.a

    ec12 = alpha12**(-1 / 4) * alpha23**(3 / 4) * alpha23**(-1 / 8) * (1 - alpha12)
    ec23 = alpha23**(-1 / 2) * alpha12**(1 / 8) * (1 - alpha23)
    w1 = np.abs((p3.n / (2*np.pi)) * (m_tot / mStar) * ((mu1 / (mu1+mu3))
                / ec12**2 + (mu3 / (mu1 + mu3)) / ec23**2))
    Tsec = 2 * np.pi / w1
    # normalize secular timescale to be in terms of 
    # number of orbits of inner most planet
    return Tsec / minP


 ######################### Taken from celmech github.com/shadden/celmech
def farey_sequence(n):
    """Return the nth Farey sequence as order pairs of the form (N,D) where `N' is the numerator and `D' is the denominator."""
    a, b, c, d = 0, 1, 1, n
    sequence=[(a,b)]
    while (c <= n):
        k = int((n + b) / d)
        a, b, c, d = c, d, (k*c-a), (k*d-b)
        sequence.append( (a,b) )
    return sequence
def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
    """Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    if min_per_ratio < 0.:
        raise AttributeError("min_per_ratio of {0} passed to resonant_period_ratios can't be < 0".format(min_per_ratio))
    if max_per_ratio >= 1.:
        raise AttributeError("max_per_ratio of {0} passed to resonant_period_ratios can't be >= 1".format(max_per_ratio))
    minJ = int(np.floor(1. / (1. - min_per_ratio)))
    maxJ = int(np.ceil(1. / (1. - max_per_ratio)))
    res_ratios=[(minJ-1,minJ)]
    for j in range(minJ,maxJ):
        res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
    res_ratios = np.array(res_ratios)
    msk = np.array( list(map( lambda x: min_per_ratio < x[0] / float(x[1]) < max_per_ratio , res_ratios )) )
    return res_ratios[msk]
##########################

def get_pairs(sim, trio):
    ''' 
    returns the three pairs of the given trio sorted by 2BRfill factor.
    
    Arguments:
        sim: simulation in question
        trio: indices of the 3 particles in question, formatted as [p1, p2, p3]
    return: returns the two pairs in question, formatted based on the magnitude of 
            the two body filling factor between said pair
                [[Max 2BR fill, index, index], [Min 2BR fill, index, index]]
    '''
 
    ps = sim.particles
    # sort trio from inner to outer
    sortedIndices = sorted(trio, key = lambda i: ps[i].a) 
    
    a,b,c = sortedIndices

    # Calculate the eccentricity vectors
    eveca = ps[a].e*np.exp(1j*ps[a].pomega)
    evecb = ps[b].e*np.exp(1j*ps[b].pomega)
    evecc = ps[c].e*np.exp(1j*ps[c].pomega)

    # Calculate relative eccentricity magnitude

    EMab = np.abs(evecb - eveca)
    EMbc = np.abs(evecc - evecb)
    
    # Calculate the two body filling factor
    fillab = twoBRFillFac(ps[a].P/ps[b].P, ps[a].m / ps[0].m, ps[b].m / ps[0].m, EMab)
    fillbc = twoBRFillFac(ps[b].P/ps[c].P, ps[b].m / ps[0].m, ps[c].m / ps[0].m, EMbc)


    if fillbc < fillab:
        return [['Max', sortedIndices[0], sortedIndices[1]],
                ['Min', sortedIndices[1], sortedIndices[2]]]
    else:
        return [['Max', sortedIndices[1], sortedIndices[2]],
                ['Min', sortedIndices[0], sortedIndices[1]]]

def hillfac(sim, i1=1, i2=2): 
    '''
    Calculates the Hill stability ratio for a two-planet system.

    Arguments:
        sim: rebound simulation
        i1: index of the inner planet
        i2: index of the outer planet
    Returns:
        hillFac: Returns sqrt(|(p/a)/(p/a)_crit}) (see Gladman 1993 and Marchal and Bozis 1982). 

        If hillFac > 1: stable (close approaches are not allowed). If hillFac < 1: unstable 
        (close approaches are allowed). Note that hillfac tends to get smaller the closer the orbits are, 
        but if planets get inside their mutual Hill spheres, the contribution of the potential energy makes hillfac > 1 again.
    '''
    ps = sim.particles
    m0 = ps[0].m  #star 
    m1 = ps[i1].m
    m2 = ps[i2].m

    M = m1 + m2 + m0  # total system mass 

    M_prod = m1*m2 + m1*m0 + m2*m0

    G = sim.G # gravitational constant

    c = np.linalg.norm(sim.angular_momentum())
    h = sim.energy()
    # calculate p/a from Eq. 12 of Gladman 1993
    pOvera = - ((2*M) / (G**2 * M_prod**3)) * c**2 * h

    # calculate p/a_crit -1 from Eq. 13, always > 0
    pOveraCritMinus1 = ((3**(4/3)) * (m1 * m2)) / (m0**(2/3) * (m1 + m2)**(4/3))

    # ignore remaining terms since it only gets smaller
    sign = np.sign(pOvera-1) # if this is negative we always fail Hill criterion

    # hillFac reduces to Delta/Deltacrit in the limit of low-mass planets on coplanar, circular orbits

    hillFac = sign*(abs((pOvera - 1)) / pOveraCritMinus1)**(1/2)

    return hillFac

# modified from original spock, some comments changed
####################################################
def get_resonance_window(pratio):
    """
    Returns minperiodratio and maxperiodratio resonances to test.
    """
    small = 1e-10
    if pratio > 0.5:
        jmin = 1
        jmax = 100000

        while jmax - jmin > 1:
            jtest = (jmin + jmax) // 2
            test_ratio = jtest / (jtest + 1)

            if pratio < test_ratio:
                jmax = jtest
            else:
                jmin = jtest

        minperiodratio = jmin / (jmin + 1) - small
        maxperiodratio = jmax / (jmax + 1) + small

    else: # resonances below 1/2
        minperiodratio = 0
        maxperiodratio = 0.5 + small

    return minperiodratio, maxperiodratio


# from https://arxiv.org/pdf/2410.21748
Aq_values = [None, 0.845, 0.754, 0.748, 0.778, 0.832, 0.904, 0.995, 1.104, 1.235, 1.388]

def find_strongest_MMR_width(sim, i1, i2, maxorder = 5):
    """
    Calculates the normalized, actual resonant width (Δ) and normalized, maximum resonance width (Δ_max) 
    for the strongest resonance between planets i1 and i2. 
    Uses the Δ_max equation and assumes best case where θ = π.

    Arguments:
        sim: rebound simulation
        i1: index of the inner planet
        i2: index of the outer planet

    Returns:
        j: if system is 2:3, j=3
        k: order of resonance (if system is 2:3, k = 1)
        min_ratio: The lowest ratio will be the system with the strongest resonance
                   Calculate as normalized width of system (Δ) / normalized max resonance width (Δ_max)
    """
    ps = sim.particles
    p1 = ps[i1]
    p2 = ps[i2]

    # sort by semi-major axis
    if p1.a < p2.a:
        inner, outer = p1, p2
    else:
        inner, outer = p2, p1

    # period ratio 
    Pratio_actual = inner.P / outer.P
    if Pratio_actual < 0 or Pratio_actual > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan

    minperiodratio, maxperiodratio = get_resonance_window(Pratio_actual)
    if np.isnan(minperiodratio) or np.isnan(maxperiodratio):
        return np.nan, np.nan, np.nan
    res = resonant_period_ratios(minperiodratio, maxperiodratio, order=maxorder)
    
    # calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega))**2 + 
                 (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega))**2)
    

    # calculation for delta_max
    ec = (outer.a - inner.a) / outer.a # valid in assumption ((delta a)/a) << 1
    mu = (outer.m + inner.m) / ps[0].m

    min_p, min_q, min_ratio = np.nan, np.nan, np.inf

    for a, b in res:
        q = b - a  # resonance order
        
        Pratio_res = a / b
        delta_actual = abs(Pratio_actual - Pratio_res) / Pratio_res

        Aq = Aq_values[q]
        delta_max = 3 * Aq * np.sqrt(mu * (EM / ec) ** q)

        if delta_max == 0 or np.isnan(delta_max):
            min_ratio = np.nan

        ratio = delta_actual / delta_max
        if ratio < min_ratio:
            min_p = b
            min_q = b - a
            min_ratio = ratio

    if min_ratio == np.inf:
        min_ratio = np.nan

    return min_p, min_q, min_ratio
##############################################

def swap(a, b):
    '''Simple swap function'''
    return b, a

def twoBRFillFac(pRat, mu1, mu2, EM):
    '''Calculates the two body resonance overlap filling factor for a given pair.
        Derived in Hadden 2018
        Param:
            pRat: the period ratio of the two planets in question
            mu1: the mass ratio of inner planet and sun
            mu2: the mass ratio of outer planet and sun
            EM: the combined eccentricity
    '''
    # uses periods instead of semimajor axis
    # checking for nan
    if pRat != pRat or EM != EM or pRat <= 0.5 or pRat >=1:
        # if ratio is less then 1/2 then there is no first order res that is near
        # if pRat >=1 it means something is wrong
        # if ratio or EM are nan something is wrong
        return np.nan

    orderConsider = 4 # up to what order to consider
    

    #first we will find the first order res on either side
    firstBelow = 0
    firstAbove = 0
    o = 1
    while (firstBelow == 0 or firstAbove == 0):
        # while we have not found the adjacent first order resonances,
        # look for them
        rat = o/(o+1)
        if rat <= pRat and rat > firstBelow:
            firstBelow = rat
        elif rat > pRat:
            firstAbove = rat
        o+=1
    # now we can generate a list of all the resonances in between along with their order
    resList = resonance_jk_list(firstBelow, firstAbove,orderConsider)
    sumVal = 0
    # we can now sum all of the resonance widths

    
    Z0 = EM
    
    for e in resList:
        minP, maxP = resonance_pratio_span(mu1, mu2, Z0, e[0], e[1])
        sumVal+= maxP - minP
    
    #now we can multiply by the normalization factor and returl

    return sumVal / (firstAbove - firstBelow)

def getIntPrat(sim, i1, i2, maxorder=5):
    """
    Returns the j,k resonance closest to the current period ratio,
    using find_strongest_MMR_width().
    """
    j, k, strength = find_strongest_MMR_width(sim, i1, i2, maxorder=maxorder)

    if np.isnan(j) or np.isnan(k):
        return None

    #convert j,k into the integer period ratio format
    p = j - k
    q = j
    
    return p, q, strength

def calcThetaVec(la, pomegaa, coefa, lb, pomegab, coefb, val,):
    theta = (val[1]*lb) - (val[0]*la) - (pomegaa * coefa) -(pomegab * coefb)
    return np.exp(theta*1j)


def calcThetaRelVec(la, lb, val, pomegarel):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.exp(theta*1j)


