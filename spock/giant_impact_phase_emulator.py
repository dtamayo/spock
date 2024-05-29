import numpy as np
import random
import os
import torch
import time
import warnings
import rebound as rb
from spock import DeepRegressor
from spock import CollisionOrbitalOutcomeRegressor, CollisionMergerClassifier
from .simsetup import copy_sim, align_simulation, get_rad, perfect_merge, replace_trio, revert_sim_units
from .tseries_feature_functions import get_collision_tseries

# planet formation simulation model
class GiantImpactPhaseEmulator():
    # initialize function
    def __init__(self, sims, tmax=None, seed=None):
        # set random seed
        if not seed is None:
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # load classification and regression models
        self.class_model = CollisionMergerClassifier()
        self.reg_model = CollisionOrbitalOutcomeRegressor()

        # load SPOCK model
        self.deep_model = DeepRegressor()
            
        if type(sims) != list:
            sims = [sims] # single simulation

        self.original_units = sims[0].units # record units of original sims
        self.original_G = sims[0].G # record value of G for input sims
        
        self.original_Mstars = np.zeros(len(sims))
        self.original_a1s = np.zeros(len(sims))
        self.original_P1s = np.zeros(len(sims))
        for i, sim in enumerate(sims):
            self.original_Mstars[i] = sim.particles[0].m
            self.original_a1s[i] = sim.particles[1].a
            self.original_P1s[i] = sim.particles[1].P
        
        self.sims = [copy_sim(sim, np.arange(1, sim.N), scaled=True) for sim in sims]
        self.orbsmax = self._get_orbsmax(tmax) # initialize array of max orbits for each simulation (default 1e9)

    # main function: predict giant impact outcomes, stop once all trios have t_inst < tmax (tmax has the same units as sim.t) 
    def predict(self, reuse_inputs=True):
        while np.any([sim.t < self.orbsmax[i] for i, sim in enumerate(self.sims)]): # take another step if any sims are still at t < tmax
            self.step(original_units=False, reuse_inputs=reuse_inputs) # keep dimensionless units until the end
               
        # convert units back to original units for final sims
        return revert_sim_units(self.sims, self.original_Mstars, self.original_a1s, self.original_G, self.original_units, self.original_P1s)

    # take another step in the iterative process, merging any planets that go unstable with t_inst < tmax
    def step(self, original_units=True, reuse_inputs=True):
        for i, sim in enumerate(self.sims): # assume all 2 planet systems (N=3) are stable (could use Hill stability criterion)
            if sim.N < 4:
                sim.t = self.orbsmax[i]

        sims_to_update = [sim for i, sim in enumerate(self.sims) if sim.t < self.orbsmax[i]]
        # estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            return

        t_insts, trio_inds = self.get_unstable_trios(sims_to_update)

        # get list of sims for which planets need to be merged
        sims_to_merge = []
        trios_to_merge = []
        instability_times = []
        for i, sim in enumerate(sims_to_update):
            idx = self.sims.index(sim)      # get index in original list
            if t_insts[i] > self.orbsmax[idx]:   # won't merge before max orbits, so update to that time
                self.sims[idx].t = self.orbsmax[idx]
            else:
                sims_to_merge.append(sim)
                trios_to_merge.append(trio_inds[i])
                instability_times.append(t_insts[i])

        # get new sims with planets merged
        self.handle_mergers(sims_to_merge, trios_to_merge, instability_times, reuse_inputs=reuse_inputs)
       
        # convert units back to original units for final sims
        if original_units:
            self.sims = revert_sim_units(self.sims, self.original_Mstars, self.original_a1s, self.original_G, self.original_units, self.original_P1s)

    # get unstable trios for list of sims using SPOCK deep model
    def get_unstable_trios(self, sims):
        trio_sims = []
        trio_inds = []
        Npls = [sim.N - 1 for sim in sims]
        
        # break system up into three-planet sub-systems
        for i in range(len(sims)):
            for j in range(Npls[i] - 2):
                trio_inds.append([j+1, j+2, j+3])
                trio_sims.append(copy_sim(sims[i], [j+1, j+2, j+3]))

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

    def handle_mergers(self, sims, trio_inds, updated_times, reuse_inputs=True):
        # predict collision probabilities with classification model
        if reuse_inputs:
            pred_probs, mlp_inputs, thetas, done_inds = self.class_model.predict_collision_probs(sims, trio_inds, return_inputs=reuse_inputs)
        else:
            pred_probs = self.class_model.predict_collision_probs(sims, trio_inds, return_inputs=reuse_inputs)

        # sample predicted probabilities
        rand_nums = np.random.rand(len(pred_probs))
        collision_inds = np.zeros((len(pred_probs), 2))
        for i, rand_num in enumerate(rand_nums):
            if rand_num < pred_probs[i][0]:
                collision_inds[i] = [1, 2]
            elif rand_num < pred_probs[i][0] + pred_probs[i][1]:
                collision_inds[i] = [2, 3]
            else:
                collision_inds[i] = [1, 3]
        
        # predict post-collision orbital states with regression model
        if reuse_inputs:
            new_sims = self.reg_model.predict_collision_outcome(sims, trio_inds, collision_inds, mlp_inputs=mlp_inputs, thetas=thetas, done_inds=done_inds)
        else:
            new_sims = self.reg_model.predict_collision_outcome(sims, trio_inds, collision_inds)
        
        # update sims
        for i, sim in enumerate(sims):
            idx = self.sims.index(sim) # find index in original list
            self.sims[idx] = new_sims[i]
            self.sims[idx].t = updated_times[i]
    
    # internal function with logic for initializing orbsmax as an array and checking for warnings
    def _get_orbsmax(self, tmax):
        if tmax:    # used passed value
            try:
                len(tmax) == len(self.sims)
            except:
                tmax = tmax*np.ones(len(self.sims)) # convert from float to array

            orbsmax = tmax/self.original_P1s   
        else:       # default = 1e9 orbits
            orbsmax = 1e9*np.ones(len(self.sims))

        if np.any([Norbs > 1.01e9 for Norbs in orbsmax]): # leave fudge factor for unit conversions
            for i, Norbs in enumerate(self.orbsmax):
                if Norbs > 1.01e9:
                    warnings.warn('Giant impact phase emulator not trained to predict beyond 10^9 orbits, check results carefully (tmax for sim {0} = {1} = {2} orbits)'.format(i, orbsmax[i]*self.original_P1s[i], orbsmax[i]))
        return orbsmax
