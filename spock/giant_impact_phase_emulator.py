import numpy as np
import random
import os
import torch
import time
import warnings
import rebound
from spock import DeepRegressor
from spock import CollisionOrbitalOutcomeRegressor, CollisionMergerClassifier
from .simsetup import sim_subset

# planet formation simulation model
class GiantImpactPhaseEmulator():
    # initialize function
    def __init__(self, seed=None):
        # set random seed
        if not seed is None:
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # load classification and regression models
        self.class_model = CollisionMergerClassifier()
        self.reg_model = CollisionOrbitalOutcomeRegressor()
        self.deep_model = DeepRegressor()
            
    # main function: predict giant impact outcomes, stop once all trios have t_inst < tmax (tmax has the same units as sim.t) 
    def predict(self, sims, tmaxs=None):
        sims, tmaxs = self._make_lists(sims, tmaxs)
        while np.any([sim.t < tmaxs[i] for i, sim in enumerate(sims)]): # take another step if any sims are still at t < tmax
            sims = self.step(sims, tmaxs) # keep dimensionless units until the end
               
        return sims

    # take another step in the iterative process, merging any planets that go unstable with t_inst < tmax
    # tmax can't be optional because if you call step over and over, 1e9 orbit default changes if inner planet merges 
    def step(self, sims, tmaxs): 
        sims, tmaxs = self._make_lists(sims, tmaxs)
        for i, sim in enumerate(sims): # assume all 2 planet systems (N=3) are stable (could use Hill stability criterion)
            if sim.N < 4:
                sim.t = tmaxs[i]

        sims_to_update = [sim for i, sim in enumerate(sims) if sim.t < tmaxs[i]]
        # estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            return sims

        t_insts, trio_inds = self._get_unstable_trios(sims_to_update)
        
        # get list of sims for which planets need to be merged
        sims_to_merge = []
        trios_to_merge = []
        for i, sim in enumerate(sims_to_update):
            idx = sims.index(sim)           # get index in original list
            if t_insts[i] > tmaxs[idx]:     # won't merge before max orbits, so just update to that time
                sims[idx].t = tmaxs[idx]
            else:                           # need to merge
                sim.t += t_insts[i]         # update time
                sims_to_merge.append(sim)
                trios_to_merge.append(trio_inds[i])
        # get new sims with planets merged
        sims = self._handle_mergers(sims, sims_to_merge, trios_to_merge)
        return sims

    # get unstable trios for list of sims using SPOCK deep model
    def _get_unstable_trios(self, sims):
        trio_sims = []
        trio_inds = []
        Npls = [sim.N - 1 for sim in sims]
        
        # break system up into three-planet sub-systems
        for i in range(len(sims)):
            for j in range(Npls[i] - 2):
                trio_inds.append([j+1, j+2, j+3])
                trio_sims.append(sim_subset(sims[i], [j+1, j+2, j+3]))

        # predict instability times
        t_insts, _, _ = self.deep_model.predict_instability_time(trio_sims, samples=1)

        # get the minimum sub-trio instability time for each system
        min_t_insts = []
        min_trio_inds = []
        for i in range(len(sims)):
            temp_t_insts = []
            temp_trio_inds = []
            for j in range(Npls[i] - 2):
                temp_t_insts.append(t_insts[int(np.sum(Npls[:i]) - 2*i + j)])
                temp_trio_inds.append(trio_inds[int(np.sum(Npls[:i]) - 2*i + j)])
            min_ind = np.argmin(temp_t_insts)
            min_t_insts.append(temp_t_insts[min_ind])
            min_trio_inds.append(temp_trio_inds[min_ind])

        return min_t_insts, min_trio_inds

    def _handle_mergers(self, sims, sims_to_merge, trio_inds):
        # predict which planets will collide in each trio_sim
        collision_inds = self.class_model.sample_collision_indices(sims_to_merge, trio_inds)

        # predict post-collision orbital states with regression model
        new_sims = self.reg_model.predict_collision_outcome(sims_to_merge, trio_inds, collision_inds)
       
        # update sims
        for i, sim in enumerate(sims_to_merge):
            idx = sims.index(sim) # find index in original list
            sims[idx] = new_sims[i]
        return sims

    # internal function with logic for initializing orbsmax as an array and checking for warnings
    def _make_lists(self, sims, tmaxs):
        if isinstance(sims, rebound.Simulation): # passed a single sim
            sims = [sims]
        
        if tmaxs:    # used passed value
            try:
                len(tmaxs) == len(sims)
            except:
                tmaxs = tmaxs*np.ones(len(sims)) # convert from float to array
        else:       # default = 1e9 orbits
            tmaxs = [1e9*sim.particles[1].P for sim in sims]

        for i, t in enumerate(tmaxs):
            orbsmax = t/sims[i].particles[1].P
            if orbsmax > 1.01e9:
                warnings.warn('Giant impact phase emulator not trained to predict beyond 10^9 orbits, check results carefully (tmax for sim {0} = {1} = {2} orbits)'.format(i, t, orbsmax))
        return sims, tmaxs