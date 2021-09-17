import rebound
import numpy as np
from celmech.secular import LaplaceLagrangeSystem
from celmech import Poincare
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from itertools import combinations
from .simsetup import init_sim_parameters

def eminus_max(lsys, Lambda, i1, i2):
    if i1 > i2:
        raise AttributeEror("i2 must be exterior body")
    res = {}
    res['L10'] = Lambda[i1-1]
    res['L20'] = Lambda[i2-1]
    res['L12'] = res['L10'] + res['L20']
    res['Lr'] = res['L10']*res['L20']/res['L12']
    res['psi'] = np.pi + np.arcsin(np.sqrt(res['L10']/res['L12']))
    res['R'] = np.identity(Lambda.shape[0]) # Nplanets x Nplanets rotation matrix
    # Only rotate the 2x2 submatrix corresponding to i1 and i2 with rotation matrix
    res['R'][np.ix_([i1-1,i2-1],[i1-1, i2-1])] = np.array([[np.cos(res['psi']), -np.sin(res['psi'])], [np.sin(res['psi']), np.cos(res['psi'])]])

    Mnorm = res['R'] @ lsys.Neccentricity_matrix @ res['R'].T
    Fx0 = res['R'] @ lsys.kappa0_vec/res['Lr']**(1/2)
    Fy0 = res['R'] @ lsys.eta0_vec/res['Lr']**(1/2)

    vals,T = np.linalg.eigh(Mnorm)

    Ax0 = T.T @ Fx0
    Ay0 = T.T @ Fy0
    A = np.sqrt(Ax0**2 + Ay0**2) # Mode amplitudes (A[0], A[1])
    Fmax = np.abs(T) @ A.T
    return Fmax[i1-1] # i1-1 index is eminus for the i1,i2 pair

def calc_tau_pair(sim, lsys, Lambda, i1, i2, LL_modulation=True):
    '''
    Calculates optical depth tau of MMRs between single pair of planets with index i1 and i2
    If LL_modulation is True, it will use the maximum anti-aligned eccentricity along 
    the Laplace-Lagrange secular cycle
    If False, it will use the initial value of the anti-aligned eccentricity
    '''
    if i1 > i2:
        raise AttributeEror("i2 must be exterior body")
    ps = sim.particles
    delta = (ps[i2].a-ps[i1].a)/ps[i2].a/(ps[i1].m+ps[i2].m)**(1/4)
    if LL_modulation == True: # calculate maximum along Laplace Lagrange cycle
        emax = eminus_max(lsys, Lambda, i1, i2)
    else: # use initial value of eminus
        emx = ps[i2].e*np.cos(ps[i2].pomega) - ps[i1].e*np.cos(ps[i1].pomega)
        emy = ps[i2].e*np.sin(ps[i2].pomega) - ps[i1].e*np.sin(ps[i1].pomega)
        emax = np.sqrt(emx**2 + emy**2)
    ec = (1-(ps[i1].P/ps[i2].P)**(2/3))
    tau = (1.8/delta)**2/np.abs(np.log(emax/ec))**(3/2)
    return tau
    
def calc_tau_pairs(sim, indexpairs, LL_modulation=True):
    '''
    Calculates total tau from the passed set of indexpairs, e.g. [[1,2], [2,3], [2,4]]
    If LL_modulation is True, it will use the maximum anti-aligned eccentricity along 
    the Laplace-Lagrange secular cycle
    If False, it will use the initial value of the anti-aligned eccentricity
    '''
    if np.isnan(sim.dt): # init_sim_parameters sets timestep to nan if any orbit is hyperbolic. Return tau=inf, i.e. chaotic/unstable
        tau = np.inf 
        return tau

    if LL_modulation == True:
        lsys = LaplaceLagrangeSystem.from_Simulation(sim)
        pvars = Poincare.from_Simulation(sim)
        Lambda = np.array([p.Lambda for p in pvars.particles[1:]])
    else:
        lsys = None
        Lambda = None

    tau = 0
    for i1, i2 in indexpairs:
        tau += calc_tau_pair(sim, lsys, Lambda, i1, i2, LL_modulation=LL_modulation)
    return tau

def calc_tau(sim):
    '''
    Calculates tau for each planet using adjacent neighbors, taking the maximum eminus over the Laplace-Lagrange secular cycle.
    Returns the maximum tau among the values calculated for each of the planets.
    '''
    if np.isnan(sim.dt): # init_sim_parameters sets timestep to nan if any orbit is hyperbolic. Return tau=inf, i.e. chaotic/unstable
        tau = np.inf 
        return tau
    lsys = LaplaceLagrangeSystem.from_Simulation(sim)
    pvars = Poincare.from_Simulation(sim)
    Lambda = np.array([p.Lambda for p in pvars.particles[1:]])

    tau_max = 0
    for i in range(1, sim.N_real):
        tau = 0
        if i-1 >= 1:
            tau += calc_tau_pair(sim, lsys, Lambda, i-1, i, LL_modulation=True)
        if i+1 < sim.N_real:
            tau += calc_tau_pair(sim, lsys, Lambda, i, i+1, LL_modulation=True)
        if tau > tau_max:
            tau_max = tau
    return tau_max


class AnalyticalClassifier():
    def __init__(self):
        pass
    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_tau(self, sim, n_jobs=-1):
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
        Nsims = len(sim)

        args = []
        for s in sim:
            s = s.copy()
            init_sim_parameters(s)
            minP = np.min([p.P for p in s.particles[1:s.N_real]])
            self.check_errors(s)
            args.append(s)

        if len(args) == 1: # single sim
            tau = calc_tau(args[0])    # stable will be 0 if an orbit is hyperbolic
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            pool = ThreadPool(n_jobs)
            tau = pool.map(calc_tau, args)
            tau = np.array(tau)
            pool.terminate()
            pool.join()
        return tau 

    def predict_stable(self, sim, n_jobs=-1):
        """
        Predict whether passed simulation will be stable over 10^9 orbits of the innermost planet.

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test
        n_jobs (int):               Number of cores to use for calculation (only if passing more than one simulation). Default: Use all available cores. 

        Returns:

        float:  Estimated probability of stability. Will return exactly zero if configuration goes 
                unstable within first 10^4 orbits.

        """
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
        Nsims = len(sim)
        
        tau = self.predict_tau(sim, n_jobs=n_jobs)
        prob = np.maximum(1-tau, 0) 
        
        return prob
