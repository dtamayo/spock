import rebound
import numpy as np
from celmech.secular import LaplaceLagrangeSystem
from celmech import Poincare

class AnalyticalClassifier():
    def __init__(self):
        pass
    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def _eminus_max(self, lsys, Lambda, i1, i2):
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

    def _calc_tau_pair(self, sim, lsys, Lambda, i1, i2):
        if i1 > i2:
            raise AttributeEror("i2 must be exterior body")
        ps = sim.particles
        delta = (ps[i2].a-ps[i1].a)/ps[i2].a/(ps[i1].m+ps[i2].m)**(1/4)
        emax = self._eminus_max(lsys, Lambda, i1, i2)
        ec = (1-(ps[i1].P/ps[i2].P)**(2/3))
        tau = (1.8/delta)**2/np.abs(np.log(emax/ec))**(3/2)
        return tau

    def _calc_tau(self, sim):
        lsys = LaplaceLagrangeSystem.from_Simulation(sim)
        pvars = Poincare.from_Simulation(sim)
        Lambda = np.array([p.Lambda for p in pvars.particles[1:]])

        tau_max = 0
        for i in range(1, sim.N_real):
            tau = 0
            if i-1 >= 1:
                tau += self._calc_tau_pair(sim, lsys, Lambda, i-1, i)
            if i+1 < sim.N_real:
                tau += self._calc_tau_pair(sim, lsys, Lambda, i, i+1)
            if tau > tau_max:
                tau_max = tau
        return tau_max

    def calc_tau(self, sim):
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
        Nsims = len(sim)
        taus = [self._calc_tau(s) for s in sim]

        if Nsims == 1:
            return taus[0]
        else:
            return taus

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
        
        tau = self.calc_tau(sim)
        prob = np.maximum(1-tau, 0) 
        
        return prob
