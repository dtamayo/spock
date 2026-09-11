from collections import OrderedDict
from spock.ClassifierSeries import getsecT
from celmech.resonances import resonance_pratio_span
from celmech.resonances import resonance_jk_list
import numpy as np
import math
import rebound
MAXORDER = 4

class Trio:
    def __init__(self, trio, sim):
        '''initializes new set of features for the trio.
        
            trio: indicies of the 3 particles in question, formatted as [p1, p2, p3]
            sim:  REBOUND simulation
            note: each list of the key is the series of data points, 
                  second dict is for final features
        '''
        # We keep track of the this trio and the adjacent pairs
        self.trio = trio
        self.pairs = get_pairs(sim, trio)
        
        # initialize running list which keeps track of data during simulation
        self.runningList = OrderedDict()
        self.runningList['time'] = []
        self.runningList['MEGNO'] = []
        self.runningList['threeBRfill'] = []
        self.runningList['massFac'] = []
        self.runningList['e0'] = [] # innermost planet in trio w/ index trio[0]
        self.runningList['e1'] = [] # middle planet in trio w/ index trio[1]
        self.runningList['e2'] = [] # outer most planet in trio w/ index trio[2]

        #testing the MASSFACS
        self.runningList['massFacLargeMass'] = []


        # initialize conjunction angle information

        self.theta = OrderedDict()
        
        #initiate running lists and calculate the order
        for [label, i1, i2] in self.pairs:
            self.runningList['EM' + label] = []
            self.runningList['EP' + label] = []
            self.runningList['MMRstrength' + label] = []
            self.runningList['pRat' + label] = []
            self.runningList['mu1' + label] = []
            self.runningList['mu2' + label] = []
            self.runningList['dRatio' + label] = []
            self.runningList['hillFac' + label] = []

            self.theta['order' + label] = []
            self.theta['vector' + label] = []
            self.theta['pRatio' + label] = []
            self.theta['relvector' + label] = []
        
        

    #returned features
        self.features = OrderedDict()

        for [label, i1, i2] in self.pairs:
            self.features['EMcross' + label] = np.nan
            self.features['EMfracstd' + label] = np.nan
            self.features['EPstd' + label] = np.nan
            self.features['MMRstrength' + label] = np.nan
            self.features['conjunctionMag' + label] = np.nan
            self.features['relConjunctionMag' + label] = np.nan
            self.features['pomega1Theta' + label] = np.nan
            self.features['pomega2Theta' + label] = np.nan
            self.features['p' + label] = np.nan
            self.features['q' + label] = np.nan
            self.features['dRatio' + label] = np.nan
            self.features['hillFac' + label] = np.nan
        self.features['MEGNO'] = np.nan
        self.features['MEGNOstd'] = np.nan
        self.features['massOrder'] = np.nan
        self.features['threeBRfillfac'] = np.nan
        self.features['massFac'] = np.nan
        self.features['mass_fac_lowest_std'] = np.nan
        self.features['mass_fac_lowest_norm_std'] = np.nan

        #eccentricity variation
        self.features['min_e_std'] = np.nan
        self.features['min_norm_e_std'] = np.nan

        #testing the MASSFACS
        self.features['massFacLargeMass'] = np.nan


    def fillVal(self, Nout):
        '''Fills with nan values
        
            Arguments: 
                Nout: number of datasets collected
        '''
        for each in self.runningList.keys():
            self.runningList[each] = [np.nan] * Nout

    def getNum(self):
        '''Returns number of features collected as ran'''
        return len(self.runningList.keys())

    def populateData(self, sim, minP,i):
        '''Populates the runningList data dictionary for one time step.
        
            Note: must specify how each feature is calculated and added
        '''
        ps = sim.particles

        i1, i2, i3 = self.trio # indices for the innermost, middle and outer planet in trio
        self.runningList['e0'][i] = ps[i1].e
        self.runningList['e1'][i] = ps[i2].e
        self.runningList['e2'][i] = ps[i3].e

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
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength' + label][i] = MMRs[2]

            # save mass ratios and integer period ratios
            self.runningList['mu1' + label][i] = m1 / ps[0].m
            self.runningList['mu2' + label][i] = m2 / ps[0].m
            self.runningList['pRat' + label][i] = ps[i1].P / ps[i2].P
            
            #calculate 3brfill
            self.runningList['threeBRfill'][i]= threeBRFillFac(sim, self.trio)

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

            #calculate the lowest delta ratio
            dRatios = find_strongest_MMR_width(sim, i1, i2)
            self.runningList['dRatio' + label][i] = dRatios[2]

            #calculate hillFac
            hillFacs = hillFac(sim, i1, i2)
            self.runningList['hillFac' + label][i] = hillFacs

            #calculate massFac (not needed bc it is calculated with initial conditions)
            self.runningList['massFac'][i]= massFac(sim, self.trio)


            #testing the MASSFACS

            self.runningList['massFacLargeMass'][i] = massFacLargestMass(sim, self.trio)

        # check rebound version, if old use .calculate_megno, otherwise use .megno, old is just version less then 4
        if float(rebound.__version__[0]) < 4:
            self.runningList['MEGNO'][i] = sim.calculate_megno()
        else:
            self.runningList['MEGNO'][i] = sim.megno()

    def startingFeatures(self, sim):
        '''Initializes, adding to the features that only depend on initial conditions'''

        ps = sim.particles
        for [label, i1, i2] in self.pairs:  
            # calculate crossing eccentricity
            self.features['EMcross' + label] = (ps[i2].a - ps[i1].a) / ps[i1].a
            pRat = getIntPrat(sim, i1, i2) #NEWWWW
            self.theta['pRatio' + label] = pRat
            self.theta['order' + label] = pRat[1] - pRat[0]
            self.theta['vector' + label] = np.zeros(pRat[1] - pRat[0] + 1, dtype = complex)
            self.theta['relvector' + label] = 0.0j
            self.theta['dRatio' + label] = None   # leave at none for now because unsure what it should be
        # calculate secular timescale and adds feature
        self.features['Tsec']= getsecT(sim, self.trio)

        massSum = 0
        massOrder = [1,2,3]
        massOrder.sort(key=lambda x: ps[x].m)
        for i in range(1,4):
            massSum += np.abs(ps[i].m - ps[massOrder[i-1]].m)

        massSum = massSum / (ps[1].m + ps[2].m + ps[3].m)
        self.features['massOrder'] = massSum   

    def fill_features(self, sim, args):
        ''' Called after short integration to fill in the final set of features 
            that are returned to the ML model.
            
            Each feature is filled depending on some combination of runningList 
            features and initial condition features
        '''
        Norbits = args[0]
        Nout = args[1]
        trios = args[2] # should not be used, already defined for this trio in self.trio
        
        
        if not np.isnan(self.runningList['MEGNO']).any(): # no nans
            # smooth last 10% to remove oscillations around 2
            self.features['MEGNO'] = np.median(
                self.runningList['MEGNO'][-(Nout // 10):]
            )

            self.features['MEGNOstd'] = np.std(
                self.runningList['MEGNO'][(Nout // 5):]
            )

        # fill features that are defined for each trio

        stds = [np.std(self.runningList['e{0}'.format(str(j))]) for j in [0,1,2]]
        NORMstds = [np.std(self.runningList['e{0}'.format(str(j))])/np.mean(self.runningList['e{0}'.format(str(j))]) for j in [0,1,2]]
        self.features['min_e_std'] = min(stds)
        self.features['min_norm_e_std'] = min(NORMstds)


        self.features['mass_fac_lowest_std'], self.features['mass_fac_lowest_norm_std'] = massFacEccentricityBoth(self.runningList, sim, self.trio)
  

        #other massFacs
        self.features['massFac']= np.median(self.runningList['massFac'])
        self.features['massFacLargeMass'] = np.median(self.runningList['massFacLargeMass'])

        # fill features that are defined for each pair iterating over all pairs
        for [label, i1, i2] in self.pairs: 
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
            
            # calculate the two body filling factor based on the avg behavior

            self.features['2BRfill' + label] = twoBRFillFac( 
                                                            np.nanmean(self.runningList['pRat' + label]),
                                                            np.nanmean(self.runningList['mu1' + label]),
                                                            np.nanmean(self.runningList['mu2' + label]),
                                                            np.nanmean(self.runningList['EM' + label])
                                                            )
            # selects the conjunction angle vector for the strongest resonance and normalizes
            self.features['conjunctionMag' + label] = np.max(np.abs(self.theta['vector' + label])) / Nout
            self.features['pomega1Theta' + label] = np.abs(self.theta['vector' + label][-1]) / Nout
            self.features['pomega2Theta' + label] = np.abs(self.theta['vector' + label][0]) / Nout
            self.features['p' + label] = np.max(self.theta['pRatio' + label])
            self.features['q' + label] = np.max(self.theta['order' + label])
            
            # calculates conjunction angle consistency based on relative pomega
            self.features['relConjunctionMag' + label] = np.abs(self.theta['relvector' + label]) / Nout
            self.features['threeBRfillfac']= np.mean(self.runningList['threeBRfill'])

            #calculates delta_max
            self.features['dRatio' + label] = np.median(
                self.runningList['dRatio' + label][1:]
            )

            # #calculates hillFac
            self.features['hillFac' + label] = np.mean(
            self.runningList['hillFac' + label][1:])

        conj_mags = [self.features['conjunctionMag' + pair[0]] for pair in self.pairs]
        self.features['conjunctionMagCutoff'] = np.nanmax(conj_mags)

        rel_conj_mags = [self.features['relConjunctionMag' + pair[0]] for pair in self.pairs]
        self.features['relConjunctionMagCutoff'] = np.nanmax(rel_conj_mags)

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

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label, i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis

def get_pairs(sim, trio):
    ''' returns the three pairs of the given trio.
    
    Arguments:
        sim: simulation in question
        trio: indicies of the 3 particles in question, formatted as [p1, p2, p3]
    return: returns the two pairs in question, formatted based on the magnitude of 
            the two body filling factor between said pair
                [[Max 2BR fill, index, index], [Min 2BR fill, index, index]]
    '''
 
    ps = sim.particles
    sortedindices = sorted(trio, key = lambda i: ps[i].a) # sort from inner to outer
    
    a,b,c = sortedindices

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
        return [['Max', sortedindices[0], sortedindices[1]],
                ['Min', sortedindices[1], sortedindices[2]]]
    else:
        return [['Max', sortedindices[1], sortedindices[2]],
                ['Min', sortedindices[0], sortedindices[1]]]

# taken from original spock, some comments changed
####################################################
def find_strongest_MMR(sim, i1, i2):
    '''Finds the strongest MMR between two planets

        Arguments:
            sim: the simulation in question
            i1: the inner most of the two planets in question
            i2: the outer most of the two planets in question
        return: information about the resonance, the third item (index 2)
                is the maximum strength of the resonance between planets
    '''
    maxorder = 2
    ps = sim.particles
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m / ps[0].m
    m2 = ps[i2].m / ps[0].m

    Pratio = n2 / n1

    delta = 0.03
    if Pratio < 0 or Pratio > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan

    minperiodratio = max(Pratio - delta, 0.)
    maxperiodratio = min(Pratio + delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio, maxperiodratio, order=maxorder)

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    ex = ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega)
    ey = ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega)
    
    EM = np.sqrt(ex**2 + ey**2)
    # pomega12 = np.arctan2(ey, ex) 
    
    
    EMcross = (ps[i2].a - ps[i1].a) / ps[i1].a

    j, k, maxstrength = np.nan, np.nan, 0 
    for a, b in res:
        nres = (b * n2 - a * n1) / n1
        if nres == 0:
            s = np.inf # still want to identify as strongest MMR if initial condition is exatly b*n2-a*n1 = 0
        else:
            s = np.abs(np.sqrt(m1 + m2) * (EM / EMcross)**((b - a) / 2.) / nres)
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s
    if maxstrength == 0:
        maxstrength = np.nan

    return j, k, maxstrength
##############################################



# testing new version of find_strongest_MMR

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

    # Sort by semi-major axis
    if p1.a < p2.a:
        inner, outer = p1, p2
    else:
        inner, outer = p2, p1

    # Period ratio 
    Pratio_actual = inner.P / outer.P
    if Pratio_actual < 0 or Pratio_actual > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan

    minperiodratio, maxperiodratio = get_resonance_window(Pratio_actual)
    if np.isnan(minperiodratio) or np.isnan(maxperiodratio):
        return np.nan, np.nan, np.nan
    res = resonant_period_ratios(minperiodratio, maxperiodratio, order=maxorder)
    
    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega))**2 + 
                 (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega))**2)
    

    # Calculation for delta_max
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

def hillFac(sim, i1, i2):
    '''
    Calculates the Hill stability factor ratio for a two-planet system: (p/a) / (p/a)_crit
    where p/a is the Marchal-Bozis parameterization of the system's dynamics.

    Arguments:
        sim: rebound simulation
        i1: index of the inner planet
        i2: index of the outer planet
    Returns:
        hillFac: the information necessary to determine if a two planet system is stable or 
        not.
    '''
    ps = sim.particles
    m0 = ps[0].m  #star 
    m1 = ps[i1].m
    m2 = ps[i2].m

    M = m1 + m2 + m0  # total system mass 

    M_prod = m1*m2 + m1*m0 + m2*m0 

    G = sim.G # gravitational constant

    c_vec = np.zeros(3) #total angular momentum
    KE = 0
    for j in [0, i1, i2]:
        r_j = np.array([ps[j].x, ps[j].y, ps[j].z])
        v_j = np.array([ps[j].vx, ps[j].vy, ps[j].vz])

        c_vec += ps[j].m * np.cross(r_j, v_j) #angular momentum vec

    
        v = np.linalg.norm(v_j) 
        m_j = ps[j].m 

        KE += 0.5 * m_j * (v**2) # add each particle

    c = np.linalg.norm(c_vec)

    U = 0 # potential energy initialize

    for j1, j2 in [[0, i1], [0, i2], [i1, i2]]:
        r_a = np.array([ps[j1].x, ps[j1].y, ps[j1].z])
        r_b = np.array([ps[j2].x, ps[j2].y, ps[j2].z])
        r = np.linalg.norm(r_a - r_b)
        
        m_a = ps[j1].m
        m_b = ps[j2].m

        U += - (G*m_a*m_b)/r # adding all of the potential energies     
    
    h = KE + U

    #calculate p/a from Eq. 12
    param = - ((2*M) / (G**2 * M_prod**3))* c**2 * h 
    
    #calculate p/a_crit from Eq. 13
    term1 = ((3**(4/3)) * (m1 * m2)) / (m0**(2/3) * (m1 + m2)**(4/3))
    paramCrit = 1 + term1  #  ignore remaining terms since it only gets smaller
    
    hillFac = (abs((param - 1) / (paramCrit - 1)))**(1/2)
    # if hillFac > 1, stable (close approaches are not allowed)
    # if hillFac < 1, unstable (close approaches are allowed)

    return hillFac


def massFac(sim, trio):
    '''
    Calculates the mass factor (m1 + m3) / m2 to assess whether the three-body
    system acts effectively as a two-body system involving the central planet.
    Returns:
        massFac: (m1 + m3) / m2
    '''
    ps = sim.particles
    b1, b2, b3 = ps[trio[0]], ps[trio[1]], ps[trio[2]]
    m1, m2, m3 = b1.m, b2.m, b3.m

    massFac = (m1 + m3) / m2

    return massFac


def massFacEccentricityBoth(runningList, sim, trio):
        '''
        Computes two versions of massFac: # Update this
        
        1. massFacLowVarE: Uses the planet with the smallest variation in eccentricity (std) as the denominator.
        2. massFacNormLowVarE: Uses the planet with the smallest normalized variation (std divided by mean) as the denominator.

        Returns:
            massFacLowVarE, massFacNormLowVarE
        '''
        # print(sim.t)
        lowest_std_index = -1
        lowest_std_value = np.inf
        lowest_norm_std_index = -1
        lowest_norm_std_value = np.inf
        for j, i in enumerate(trio): # j is always 0, 1, 2, i is the index (could be 2,3,4)
            e = runningList['e{0}'.format(str(j))]
            if len(e) == 0 or np.isnan(e).any():
                # Invalid data, return NaNs
                return np.nan, np.nan
            
            if np.std(e) < lowest_std_value:
                lowest_std_index = i
                lowest_std_value = np.std(e)
            if np.std(e)/np.mean(e) < lowest_norm_std_value:
                lowest_norm_std_index = i
                lowest_norm_std_value = np.std(e)/np.mean(e)

        #if it didn't change
        if lowest_std_index == -1 or lowest_norm_std_index == -1:
            return np.nan, np.nan
        
        # if lowest_std_index not in trio: raise ValueError('No planet had lowest std(e)?')
        # if lowest_norm_std_index not in trio: raise ValueError('No planet had lowest std(e)/mean(e)?')

        # calculate massFacLowVarE
        ps = sim.particles
        # print(trio)
        i1, i2, i3 = trio
        m_tot = ps[i1].m + ps[i2].m + ps[i3].m

        m_max = ps[lowest_std_index].m  # biggest mass is the one that changes the least
        m_others = m_tot - m_max
        mass_fac_lowest_std = m_others/m_max # mass of smaller two planets / biggest mass

        m_max = ps[lowest_norm_std_index].m  # biggest mass is the one that changes the least
        m_others = m_tot - m_max
        mass_fac_lowest_norm_std = m_others/m_max # mass of smaller two planets / biggest mass

        return mass_fac_lowest_std, mass_fac_lowest_norm_std



def massFacLargestMass(sim, trio):
    """
    puts the largest massed planet in the denominator 

    adds the other two in numerator
    """
    ps = sim.particles

    mass_list = []
    for i in trio:
        mass_list.append((i, ps[i].m))


    largest = mass_list[0]
    for pair in mass_list:
        if pair[1] > largest[1]:
            largest = pair

    central_idx = largest[0]
    m2 = largest[1]  # denominator

    num_indices = []
    for i in trio:
        if i != central_idx:
            num_indices.append(i)

    m1 = ps[num_indices[0]].m
    m3 = ps[num_indices[1]].m

    massFacLargeMass = (m1 + m3) / m2
    
    return massFacLargeMass

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
        return 0

    orderConsider = MAXORDER # up to what order to consider
    

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

def getIntPrat(sim, i1, i2, maxorder=4):
    """
    Returns the j,k resonance closest/strongest to the current period ratio,
    using find_strongest_MMR_width().
    """
    j, k, strength = find_strongest_MMR_width(sim, i1, i2, maxorder=maxorder)

    if np.isnan(j) or np.isnan(k):
        return None

    # Convert j,k into the integer period ratio format (e.g., for 2:3, return (2,3))
    p = j - k
    q = j
    
    return p, q, strength


def calcThetaVec(la, pomegaa, coefa, lb, pomegab, coefb, val,):
    theta = (val[1]*lb) - (val[0]*la) - (pomegaa * coefa) -(pomegab * coefb)
    return np.exp(theta*1j)


def calcThetaRelVec(la, lb, val, pomegarel):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.exp(theta*1j)




def threeBRFillFac(sim, trio):
    '''calculates the 3BR filling factor in acordance to petit20'''
    ps = sim.particles
    b0, b1,b2,b3 = ps[0], ps[trio[0]], ps[trio[1]], ps[trio[2]]
    m0,m1,m2,m3 = b0.m,b1.m,b2.m,b3.m
    ptot = None

    #semim
    a12 =(b1.a/b2.a)
    a23 = (b2.a/b3.a)

    #equation 43
    d12 = 1- a12
    d23 = 1- a23

    #equation 45
    d = (d12*d23)/(d12+d23)

    #equation 19
    mu12 = b1.P/b2.P
    mu23 = b2.P/b3.P

    #equation 21
    eta = (mu12*(1-mu23))/(1-(mu12*mu23))

    #equation 53
    eMpow2 = (m1*m3 + m2*m3*(eta**2)*(a12**(-2))+m1*m2*(a23**2)*((1-eta)**2))/(m0**2)

    #equation 59
    dov = ((42.9025)*(eMpow2)*(eta*((1-eta)**3)))**(0.125)

    #equation 60

    ptot = (dov/d)**4
    return abs(ptot)