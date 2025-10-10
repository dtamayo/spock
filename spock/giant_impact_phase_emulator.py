import numpy as np
import random
import os
import torch
import time
import warnings
import rebound as rb
from spock import CollisionOrbitalOutcomeRegressor, CollisionMergerClassifier, DeepRegressor
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
        self.class_model = CollisionMergerClassifier(seed=seed)
        self.reg_model = CollisionOrbitalOutcomeRegressor(seed=seed)
        self.deep_model = DeepRegressor()
            
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
            
        if verbose: tot_start = time.time() # record start time
        
        # main loop
        sims, tmaxs = self._make_lists(sims, tmaxs)
        while np.any([sim.t < tmaxs[i] for i, sim in enumerate(sims)]): # take another step if any sims are still at t < tmax
            sims = self.step(sims, tmaxs, verbose=verbose, deepregressor_kwargs=deepregressor_kwargs)
            if isinstance(sims, rb.Simulation): sims = [sims] # passed a single sim
               
        # print total time
        if verbose:
            tot_end = time.time()
            print('Total time:', tot_end - tot_start, 's')
        
        if len(sims) == 1:
            return sims[0] # return single sim
        
        return sims

    def step(self, sims, tmaxs, verbose=False, deepregressor_kwargs={'samples':100, 'max_model_samples':10}):
        """
        Perform another step in the iterative prediction process, merging any planets that go unstable in t < tmax.

        Parameters:

        sims (rebound.Simulation or list): Current state of the giant impact simulations.
        tmaxs (float or list): Maximum time for simulation predictions in the same units as sims.t. Doesn't take a default to help ensure user passes same values each time if several steps done in a row.
        verbose (bool): Whether or not to provide outputs during the iterative prediction process.
        deepregressor_kwargs (dict): Keyword arguments for calls to DeepRegressor.predict_instability_time (e.g., samples, max_model_samples, Ncpus).

        Returns:
        
        rebound.Simulation or list of rebound.Simulation objects: Predicted states after one step of predicting instability times and merging planets.

        """
        if isinstance(sims, rb.Simulation): sims = [sims] # passed a single sim
        
        sims, tmaxs = self._make_lists(sims, tmaxs)
        
        sims_to_update = [sim for i, sim in enumerate(sims) if sim.t < tmaxs[i]]
        # estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            if len(sims) == 1:
                return sims[0] # return single sim
            else: 
                return sims

        if verbose:
            print('Number of sims to update:', len(sims_to_update), '\n')
            print('Predicting trio instability times')
            start = time.time()
        
        t_insts, trio_inds = self._get_unstable_trios(sims_to_update, deepregressor_kwargs=deepregressor_kwargs)
        if verbose:
            end = time.time()
            print('Done:', end - start, 's' + '\n')
        
        # get list of sims for which planets need to be merged
        sims_to_merge = []
        sims_to_merge_2p = []
        trios_to_merge = []
        for i, sim in enumerate(sims_to_update):
            idx = sims.index(sim)           # get index in original list
            if t_insts[i] > tmaxs[idx]:     # won't merge before max time, so just update to that time
                sims[idx].t = tmaxs[idx]
            else:                           # need to merge
                #sim.t += t_insts[i]         # update time
                sim.t = t_insts[i]         # update time
                if sim.N == 3:
                    sims_to_merge_2p.append(sim)
                else:
                    sims_to_merge.append(sim)
                    trios_to_merge.append(trio_inds[i])
        
        if verbose:
            print('Predicting instability outcomes')
            start = time.time()
       
        # get new sims with planets merged
        sims = self._handle_2p_mergers(sims, sims_to_merge_2p)
        if sims_to_merge:
            sims = self._handle_mergers(sims, sims_to_merge, trios_to_merge)
        
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
        if trio_sims: # there might not be any 3+ planet systems to evaluate, in which case want to skip to avoid errors
            t_insts, _, _ = self.deep_model.predict_instability_time(trio_sims, **deepregressor_kwargs)
        # get the minimum sub-trio instability time for each system to store which trio indices are the least stable
        min_trio_inds = []
        for i in range(len(sims)):
            if sims[i].N < 4: # < 3 planets
                min_trio_inds.append([i for i in range(1,sims[i].N)])
            else:
                temp_t_insts = []
                temp_trio_inds = []
                for j in range(Npls[i] - 2):
                    temp_t_insts.append(t_insts[int(np.sum(Npls[:i]) - 2*i + j)])
                    temp_trio_inds.append(trio_inds[int(np.sum(Npls[:i]) - 2*i + j)])
                min_ind = np.argmin(temp_t_insts)
                min_trio_inds.append(temp_trio_inds[min_ind])

        # We want to return the most unstable trio, but the instability time corresponding to the full system, so rerun with all planets
        # Need to group according to Npl for deepregressor to run in parallel
        full_t_insts = []
        full_inds = []
        max_Npl = max(Npls)
        for Npl in range(max_Npl+1):
            subset_sims = [sim_subset(sim, np.arange(1, sim.N)) for i, sim in enumerate(sims) if Npls[i] == Npl]
            subset_inds = [i for i, sim in enumerate(sims) if Npls[i] == Npl]
            
            if len(subset_sims) > 0:
                subset_t_insts, _, _ = self.deep_model.predict_instability_time(subset_sims, **deepregressor_kwargs)
                full_t_insts.extend(subset_t_insts)
                full_inds.extend(subset_inds)
        sort_inds = np.argsort(full_inds)
        full_t_insts = np.array(full_t_insts)[sort_inds]
            
        return full_t_insts, min_trio_inds # return the most unstable trio (by itself) and predicted inst time including all planets

    # internal function for handling mergers with class_model and reg_model
    def _handle_2p_mergers(self, sims, sims_to_merge):
        new_sims = []
        for sim in sims_to_merge:
            new_sim = sim.copy()
            ps = new_sim.particles
            sum_mass = ps[1].m + ps[2].m
            mergedPlanet = (ps[1]*ps[1].m + ps[2]*ps[2].m)/sum_mass 
            mergedPlanet.m  = sum_mass
            new_sim.remove(index=2)
            new_sim.remove(index=1)
            new_sim.add(mergedPlanet)
            new_sims.append(new_sim)
        # update sims
        for i, sim in enumerate(sims_to_merge):
            idx = sims.index(sim) # find index in original list
            sims[idx] = new_sims[i]
        return sims

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
        if not tmaxs is None:
            try:
                len(tmaxs) == len(sims)
            except:
                tmaxs = tmaxs*np.ones(len(sims)) # convert from float to array
        else:       # default = 1e9 orbits
            tmaxs = [1e9*sim.particles[1].P if sim.N > 1 else 1e9 for sim in sims] # if N=1 no planets just set to 1e9

        for i, t in enumerate(tmaxs):
            if sims[i].N > 1: # otherwise ps[1].P will error, these sims will not get run anyway so OK
                orbsmax = t/sims[i].particles[1].P
                if orbsmax > 10**9.5:
                    warnings.warn('Giant impact phase emulator not trained to predict beyond 10^9 orbits, check results carefully (tmax for sim {0} = {1:3e} = {2:3e} orbits)'.format(i, t, orbsmax))
        return sims, tmaxs

    def cite(self):
        """
        Print citations to papers relevant to this model.
        """
        
        txt = """This paper made use of stability predictions from the Stability of Planetary Orbital Configurations Klassifier (SPOCK) package \\citep{spock}. Collisional evolution was simulated with the GiantImpactPhaseEmulator \\citep{giantimpact}, repeatedly using the DeepRegressor (a Bayesian neural network) to predict the time to the next dynamical instability \\citep{deepregressor}, and multilayer perceptrons (MLPs) to predict the collisional outcome \\citep{giantimpact}."""
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
}

@ARTICLE{deepregressor,
   author = {{Cranmer}, Miles and {Tamayo}, Daniel and {Rein}, Hanno and {Battaglia}, Peter and {Hadden}, Samuel and {Armitage}, Philip J. and {Ho}, Shirley and {Spergel}, David N.},
    title = "{A Bayesian neural network predicts the dissolution of compact planetary systems}",
  journal = {Proceedings of the National Academy of Science},
 keywords = {deep learning, UAT:2173, Bayesian analysis, chaos, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Artificial Intelligence, Computer Science - Machine Learning, Statistics - Machine Learning},
     year = 2021,
    month = oct,
   volume = {118},
   number = {40},
      eid = {e2026053118},
    pages = {e2026053118},
      doi = {10.1073/pnas.2026053118},
archivePrefix = {arXiv},
   eprint = {2101.04117},
primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2021PNAS..11826053C},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{giantimpact,
author = {{Lammers}, Caleb and {Cranmer}, Miles and {Hadden}, Sam and {Ho}, Shirley and {Murray}, Norman and {Tamayo}, Daniel},
        title = "{Accelerating Giant-impact Simulations with Machine Learning}",
      journal = {\apj},
     keywords = {Exoplanets, Extrasolar rocky planets, Planet formation, Planetary dynamics, 498, 511, 1241, 2173, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning},
         year = 2024,
        month = nov,
       volume = {975},
       number = {2},
          eid = {228},
        pages = {228},
          doi = {10.3847/1538-4357/ad7fe5},
archivePrefix = {arXiv},
       eprint = {2408.08873},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...975..228L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}i
}
"""
        print(txt + "\n\n\n" + bib)

