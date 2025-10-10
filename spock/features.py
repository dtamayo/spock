from collections import OrderedDict
import numpy as np
import math
import rebound


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

        # add keys here for time series that belong to each pair in the trio
        for each in ['near','far']:
            self.runningList['EM' + each] = [np.nan] * Nout
            self.runningList['EP' + each] = [np.nan] * Nout
            self.runningList['MMRstrength' + each] = [np.nan] * Nout

        # dict of features to calculate

        self.features = OrderedDict()

        # add keys here for features that belong to each pair in the trio
        for each in ['near', 'far']:
            self.features['EMcross' + each] = np.nan
            self.features['EMfracstd' + each] = np.nan
            self.features['EPstd' + each] = np.nan
            self.features['MMRstrength' + each] = np.nan

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

            self.runningList['time'][i]= sim.t/minP
            #crossing eccentricity
            self.runningList['EM'+label][i] = np.sqrt((e2x - e1x)**2 + (e2y - e1y)**2)
            #mass weighted crossing eccentricity
            self.runningList['EP'+label][i] = np.sqrt((m1 * e1x + m2 * e2x)**2 +
                                                      (m1 * e1y + m2 * e2y)**2) / (m1+m2)
            #calculate the strength of MMRs
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength' + label][i] = MMRs[2]


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

        for label in ['near', 'far']:
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

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis

def get_pairs(sim, trio):
    ''' returns the three pairs of the given trio.
    
    Arguments:
        sim: simulation in question
        trio: indicies of the 3 particles in question, formatted as [p1, p2, p3]
    return: returns the two pairs in question, formatted as 
                [[near pair, index, index], [far pair, index, index]]
    '''
 
    ps = sim.particles
    sortedindices = sorted(trio, key = lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = ((ps[sortedindices[1]].a - ps[sortedindices[0]].a)
                    / ps[sortedindices[0]].a)

    EMcrossOuter = ((ps[sortedindices[2]].a - ps[sortedindices[1]].a)
                    / ps[sortedindices[1]].a)

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]],
                ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]],
                ['far', sortedindices[0], sortedindices[1]]]

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
    EM = np.sqrt((ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega))**2 + 
                 (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega))**2)
    
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

def swap(a, b):
    '''Simple swap function'''
    return b, a
