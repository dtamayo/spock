import numpy as np
import random
import os
import torch
import time
import warnings
import rebound as rb
from spock import DeepRegressor
from spock import CollisionOrbitalOutcomeRegressor, CollisionMergerClassifier
from .simsetup import sim_subset, remove_ejected_ps

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
        self.deep_model = DeepRegressor(seed=seed)
            
    def predict(self, sims, tmaxs=None, verbose=False, deepregressor_kwargs={'samples':100, 'max_model_samples':10}):
        """
        Predict outcome of giant impact simulations up to a user-specified maximum time.

        Parameters:

        sims (rebound.Simulation or list): Initial condition(s) for giant impact simulations.
        tmaxs (float or list): Maximum time for simulation predictions in the same units as sims.t. The default is 10^9 P1, the maximum possible time.
        verbose (bool): Whether or not to provide outputs during the iterative prediction process.
        deepregressor_kwargs (dict): Keyword arguments for calls to DeepRegressor.predict_instability_time (e.g., samples, max_model_samples, Ncpus).

        Returns:
        
        rebound.Simulation or list of rebound.Simulation objects: Predicted post-giant impact states for each provided initial condition.

        """
        if isinstance(sims, rb.Simulation): sims = [sims] # passed a single sim
        sim0 = sims[0]
        print(self.deep_model.predict_instability_time(sims[0], **deepregressor_kwargs))
        print(self.deep_model.predict_instability_time(sims[0], **deepregressor_kwargs))
        
        # main loop
        sims, tmaxs = self._make_lists(sims, tmaxs)
        while np.any([sim.t < tmaxs[i] for i, sim in enumerate(sims)]): # take another step if any sims are still at t < tmax
            sims = self.step(sims, tmaxs, verbose=verbose, deepregressor_kwargs=deepregressor_kwargs)
            if isinstance(sims, rb.Simulation): sims = [sims] # passed a single sim
        
        print('before recall', sim0.N)
        print(self.deep_model.predict_instability_time(sim0, **deepregressor_kwargs))
               
        if len(sims) == 1:
            return sims[0] # return single sim
        else: 
            return sims
                
        return sims

    def step(self, sims, tmaxs=None, verbose=False, deepregressor_kwargs={'samples':100, 'max_model_samples':10}):
        """
        Perform another step in the iterative prediction process, merging any planets that go unstable in t < tmax.

        Parameters:

        sims (rebound.Simulation or list): Current state of the giant impact simulations.
        tmaxs (float or list): Maximum time for simulation predictions in the same units as sims.t. The default is the maximum time of 10^9 orbits of the innermost planet.
        verbose (bool): Whether or not to provide outputs during the iterative prediction process.
        deepregressor_kwargs (dict): Keyword arguments for calls to DeepRegressor.predict_instability_time (e.g., samples, max_model_samples, Ncpus).

        Returns:
        
        rebound.Simulation or list of rebound.Simulation objects: Predicted states after one step of predicting instability times and merging planets.

        """

        if isinstance(sims, rb.Simulation): sims = [sims] # passed a single sim
        print(self.deep_model.predict_instability_time(sims[0], **deepregressor_kwargs))
        print(self.deep_model.predict_instability_time(sims[0], **deepregressor_kwargs))
        
        sims, tmaxs = self._make_lists(sims, tmaxs)
        
        for i, sim in enumerate(sims): # assume all 2 planet systems (N=3) are stable (could use Hill stability criterion)
            if sim.N < 4:
                sim.t = tmaxs[i]
        
        sims_to_update = [sim for i, sim in enumerate(sims) if sim.t < tmaxs[i]]
        # estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            return sims

        if verbose:
            print('Predicting trio instability times')
            start = time.time()
        #print(tmaxs, deepregressor_kwargs) 
        #print(np.random.random())
        t_insts, trio_inds = self._get_unstable_trios(sims_to_update, deepregressor_kwargs=deepregressor_kwargs)
        
        if verbose:
            end = time.time()
            print('Done:', end - start, 's' + '\n')
        
        # get list of sims for which planets need to be merged
        sims_to_merge = []
        trios_to_merge = []
        for i, sim in enumerate(sims_to_update):
            idx = sims.index(sim)           # get index in original list
            if t_insts[i] > tmaxs[idx]:     # won't merge before max time, so just update to that time
                sims[idx].t = tmaxs[idx]
            else:                           # need to merge
                #sim.t += t_insts[i]        # update time
                sim.t = t_insts[i]          # update time
                sims_to_merge.append(sim)
                trios_to_merge.append(trio_inds[i])
        
        if verbose:
            print('Predicting instability outcomes')
            start = time.time()
       
        print(sims[0].t, trios_to_merge, sims[0].N)
        # get new sims with planets merged
        sims = self._handle_mergers(sims, sims_to_merge, trios_to_merge)
        print('After:', sims[0].N)
        
        if verbose:
            end = time.time()
            print('Done:', end - start, 's' + '\n')
        
        if len(sims) == 1:
            return sims[0] # return single sim
        else: 
            return sims

    # get unstable trios for list of sims using SPOCK deep model
    def _get_unstable_trios(self, sims, deepregressor_kwargs={'samples':100, 'max_model_samples':10}):
        trio_sims = []
        trio_inds = []
        Npls = [sim.N - 1 for sim in sims]
        
        # break system up into three-planet sub-systems
        for i in range(len(sims)):
            for j in range(Npls[i] - 2):
                trio_inds.append([j+1, j+2, j+3])
                trio_sims.append(sim_subset(sims[i], [j+1, j+2, j+3]))
        
        # predict instability times for sub-trios
        t_insts, _, _ = self.deep_model.predict_instability_time(trio_sims, **deepregressor_kwargs)

        # get the minimum sub-trio instability time for each system
        min_trio_inds = []
        for i in range(len(sims)):
            temp_t_insts = []
            temp_trio_inds = []
            for j in range(Npls[i] - 2):
                temp_t_insts.append(t_insts[int(np.sum(Npls[:i]) - 2*i + j)])
                temp_trio_inds.append(trio_inds[int(np.sum(Npls[:i]) - 2*i + j)])
            min_ind = np.argmin(temp_t_insts)
            min_trio_inds.append(temp_trio_inds[min_ind])

        # predict full-system instability times (systems are grouped according to Npl)
        full_t_insts = []
        full_inds = []
        max_Npl = max(Npls)
        for Npl in range(3, max_Npl+1):
            subset_sims = [sim_subset(sim, np.arange(1, sim.N)) for i, sim in enumerate(sims) if Npls[i] == Npl]
            subset_inds = [i for i, sim in enumerate(sims) if Npls[i] == Npl]
            
            if len(subset_sims) > 0:
                subset_t_insts, _, _ = self.deep_model.predict_instability_time(subset_sims, **deepregressor_kwargs)
                full_t_insts.extend(subset_t_insts)
                full_inds.extend(subset_inds)
        sort_inds = np.argsort(full_inds)
        full_t_insts = np.array(full_t_insts)[sort_inds]
            
        return full_t_insts, min_trio_inds

    # internal function for handling mergers with class_model and reg_model
    def _handle_mergers(self, sims, sims_to_merge, trio_inds):
        # predict which planets will collide in each trio_sim
        collision_inds, ML_inputs = self.class_model.predict_collision_pair(sims_to_merge, trio_inds, return_ML_inputs=True)
        
        # get collision_inds for sims that did not experience a merger
        sims_to_merge_temp, trio_sims, mlp_inputs, done_sims, done_inds = ML_inputs
        if 0 < len(done_inds):
            mask = np.ones(len(collision_inds), dtype=bool)
            mask[np.array(done_inds)] = False
            subset_collision_inds = list(np.array(collision_inds)[mask])
        else:
            subset_collision_inds = collision_inds
        ML_inputs = sims_to_merge_temp, subset_collision_inds, trio_inds, trio_sims, mlp_inputs, done_sims, done_inds
            
        # predict post-collision orbital states with regression model
        new_sims = self.reg_model._predict_collision_probs_from_inputs(*ML_inputs)
       
        # update sims
        for i, sim in enumerate(sims_to_merge):
            idx = sims.index(sim) # find index in original list
            sims[idx] = new_sims[i]
        return sims

    # internal function with logic for initializing orbsmax as an array and checking for warnings
    def _make_lists(self, sims, tmaxs):
        sims = remove_ejected_ps(sims) # remove ejected/hyperbolic particles (do here so we don't use a negative period for tmaxs)
        
        # use passed value
        if tmaxs:
            try:
                len(tmaxs) == len(sims)
            except:
                tmaxs = tmaxs*np.ones(len(sims)) # convert from float to array
        else:       # default = 1e9 orbits
            tmaxs = [1e9*sim.particles[1].P for sim in sims]

        for i, t in enumerate(tmaxs):
            orbsmax = t/sims[i].particles[1].P
            if orbsmax > 10**9.5:
                warnings.warn('Giant impact phase emulator not trained to predict beyond 10^9 orbits, check results carefully (tmax for sim {0} = {1} = {2} orbits)'.format(i, t, orbsmax))
        return sims, tmaxs
