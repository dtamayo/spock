from itertools import combinations
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import rebound
from celmech import Poincare
from celmech.secular import LaplaceLagrangeSystem
from .features import hillfac
from .simsetup import setup_sim

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
    if np.isnan(sim.dt): # setup_sim sets timestep to nan if any orbit is hyperbolic. Return tau=inf, i.e. chaotic/unstable
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

    If system has two planets, will return tau = 0 if Hill stable, tau=1 if not Hill stable.
    If system has zero or one planets, will return tau=0 always.
    '''
    if np.isnan(sim.dt): # setup_sim sets timestep to nan if any orbit is hyperbolic. Return tau=inf, i.e. chaotic/unstable
        tau = np.inf
        return tau

    Nplanets = sim.N_real-1

    if Nplanets == 0 or Nplanets == 1:
        tau = 0
        return tau # tau = 0 = not overlapped (stable)
    if Nplanets == 2:
        tau = 0 if hillfac(sim) > 1 else 1 # if hillfac > 1 hill stable, so set tau=0, else 1
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
 
    def predict_tau(self, sim, n_jobs=-1):
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
        Nsims = len(sim)

        args = []
        for s in sim:
            s = setup_sim(s)
            minP = np.min([p.P for p in s.particles[1:s.N_real]])
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

        If system has two planets, will return whether or not the system is Hill stable. 
        If system has zero or one planets, will always return stable.
        
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

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """
        
        txt = """This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. Probabilities of stability were determined using the AnalyticalClassifier \\citep{analytical}, which calculates the degree of overlap between mean motion resonances \\citep{Hadden18}, while additionally accounting for the slow expansion and contraction of these resonances due to long-term (secular) eccentricity oscillations \\citep{Yang24}."""
        bib = """
@ARTICLE{spock,
   author = {{Tamayo}, Daniel and {Cranmer}, Miles and {Hadden}, Samuel and {Rein}, Hanno and {Battaglia}, Peter and {Obertas}, Alysa and {Armitage}, Philip J. and {Ho}, Shirley and {Spergel}, David N. and {Gilbertson}, Christian and {Hussain}, Naireen and {Silburt}, Ari and {Jontof-Hutter}, Daniel and {Menou}, Kristen},
    title = "{Predicting the long-term stability of compact multiplanet systems}",
  journal = {Proceedings of the National Academy of Science},
 keywords = {machine learning, dynamical systems, UAT:498, orbital dynamics, UAT:222, Astrophysics - Earth and Planetary Astrophysics},
     year = 2020,
    month = aug,
   volume = {117},
   number = {31},
    pages = {18194-18205},
      doi = {10.1073/pnas.2001258117},
archivePrefix = {arXiv},
   eprint = {2007.06521},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2020PNAS..11718194T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        
@ARTICLE{analytical,
   author = {{Tamayo}, Daniel and {Murray}, Norman and {Tremaine}, Scott and {Winn}, Joshua},
    title = "{A Criterion for the Onset of Chaos in Compact, Eccentric Multiplanet Systems}",
  journal = {\aj},
 keywords = {Exoplanets, Planetary dynamics, Orbital resonances, 498, 2173, 1181, Astrophysics - Earth and Planetary Astrophysics, Nonlinear Sciences - Chaotic Dynamics},
     year = 2021,
    month = nov,
   volume = {162},
   number = {5},
      eid = {220},
    pages = {220},
      doi = {10.3847/1538-3881/ac1c6a},
archivePrefix = {arXiv},
   eprint = {2106.14863},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2021AJ....162..220T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Hadden18,
   author = {{Hadden}, Sam and {Lithwick}, Yoram},
    title = "{A Criterion for the Onset of Chaos in Systems of Two Eccentric Planets}",
  journal = {\aj},
 keywords = {celestial mechanics, chaos, planets and satellites: dynamical evolution and stability, Astrophysics - Earth and Planetary Astrophysics},
     year = 2018,
    month = sep,
   volume = {156},
   number = {3},
      eid = {95},
    pages = {95},
      doi = {10.3847/1538-3881/aad32c},
archivePrefix = {arXiv},
   eprint = {1803.08510},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Yang24,
   author = {{Yang}, Qing and {Tamayo}, Daniel},
    title = "{Secular Dynamics of Compact Three-planet Systems}",
  journal = {\apj},
 keywords = {Exoplanet dynamics, Perturbation methods, Exoplanets, 490, 1215, 498, Astrophysics - Earth and Planetary Astrophysics},
     year = 2024,
    month = jun,
   volume = {968},
   number = {1},
      eid = {20},
    pages = {20},
      doi = {10.3847/1538-4357/ad3af1},
archivePrefix = {arXiv},
   eprint = {2312.10031},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...968...20Y},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
        print(txt + "\n\n\n" + bib)
