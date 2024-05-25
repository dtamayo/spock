import numpy as np
import random
import os
import torch
import time
import warnings
import rebound as rb
from spock import DeepRegressor
from spock import CollisionOrbitalOutcomeRegressor, CollisionMergerClassifier
from .simsetup import copy_sim, align_simulation, get_rad, perfect_merge, npEulerAnglesTransform

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
        
        # load regression and classification models
        pwd = os.path.dirname(__file__)
        self.reg_model = CollisionOrbitalOutcomeRegressor()#torch.load(pwd + '/models/' + reg_model_file, map_location=torch.device('cpu'))
        self.class_model = CollisionMergerClassifier()#torch.load(pwd + '/models/' + class_model_file, map_location=torch.device('cpu'))

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
    def predict(self):
        while np.any([sim.t < self.orbsmax[i] for i, sim in enumerate(self.sims)]): # take another step if any sims are still at t < tmax
            self.step(original_units=False) # keep dimensionless units until the end
            
        return self._revert_sim_units()     # convert units back to original units for final sims

    # take another step in the iterative process, merging any planets that go unstable with t_inst < tmax
    def step(self, original_units=True):
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
        self.MLP_merge_planets(sims_to_merge, trios_to_merge, instability_times)
       
        if original_units:
            self.sims = self._revert_sim_units() # convert units back to original units for final sims
    
    # replace particle in sim with new state (in place)
    def replace_p(self, sim, p_ind, new_particle):
        sim.particles[p_ind].m = new_particle.m
        sim.particles[p_ind].a = new_particle.a
        sim.particles[p_ind].e = new_particle.e
        sim.particles[p_ind].inc = new_particle.inc
        sim.particles[p_ind].pomega = new_particle.pomega
        sim.particles[p_ind].Omega = new_particle.Omega
        sim.particles[p_ind].l = new_particle.l
    
    # return sim in which planet trio has been replaced with two planets
    def replace_trio(self, original_sim, trio_inds, new_state_sim, theta1, theta2):
        # change back to units of original sim
        original_a1 = original_sim.particles[int(trio_inds[0])].a
        new_state_sim.G = original_sim.G
        new_ps = new_state_sim.particles
        for i in range(1, len(new_ps)):
            new_ps[i].a = original_a1*new_ps[i].a

        ind1, ind2, ind3 = int(trio_inds[0]), int(trio_inds[1]), int(trio_inds[2])
        if len(new_ps) == 3:   
            if new_ps[1].a < new_ps[2].a:
                sorted_ps = [new_ps[1], new_ps[2]]
            else:
                sorted_ps = [new_ps[2], new_ps[1]]
                
            sim_copy = original_sim.copy()
            self.replace_p(sim_copy, ind1, sorted_ps[0])
            self.replace_p(sim_copy, ind2, sorted_ps[1])
            sim_copy.remove(ind3)
        if len(new_ps) == 2:
            sim_copy = original_sim.copy()
            self.replace_p(sim_copy, ind1, new_ps[1])
            sim_copy.remove(ind3)
            sim_copy.remove(ind2)
        if len(new_ps) == 1:
            sim_copy = original_sim.copy()
            sim_copy.remove(ind3)
            sim_copy.remove(ind2)
            sim_copy.remove(ind1)
        
        # change axis orientation back to original sim here
        for p in sim_copy.particles[:sim_copy.N_real]:
            p.x, p.y, p.z = npEulerAnglesTransform(p.xyz, -theta1, -theta2, 0)
            p.vx, p.vy, p.vz = npEulerAnglesTransform(p.vxyz, -theta1, -theta2, 0)
        
        # re-order particles in ascending semi-major axis
        ps = sim_copy.particles
        semi_as = []
        for i in range(1, len(ps)):
            semi_as.append(ps[i].a)
        sort_inds = np.argsort(semi_as)

        ordered_sim = sim_copy.copy()
        for i, ind in enumerate(sort_inds):
            self.replace_p(ordered_sim, i+1, ps[int(ind)+1])

        return ordered_sim

    # run short sim to get input for MLP model (returns a sim if merger/ejection occurs)
    def generate_input(self, sim, trio_inds):
        # get three-planet sim
        trio_sim = copy_sim(sim, trio_inds, scaled=True)
        ps = trio_sim.particles
            
        # align z-axis with direction of angular momentum
        theta1, theta2 = align_simulation(trio_sim)

        # assign planet radii
        for i in range(1, len(ps)):
            ps[i].r = get_rad(ps[i].m)

        # set integration settings
        trio_sim.integrator = 'mercurius'
        trio_sim.collision = 'direct'
        trio_sim.collision_resolve = perfect_merge
        Ps = np.array([p.P for p in ps[1:len(ps)]])
        es = np.array([p.e for p in ps[1:len(ps)]])
        minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
        trio_sim.dt = 0.05*minTperi

        times = np.linspace(0.0, 1e4, 100)
        states = [np.log10(ps[1].m), np.log10(ps[2].m), np.log10(ps[3].m)]
        
        for t in times:
            trio_sim.integrate(t, exact_finish_time=0)

            # check for ejected planets
            if len(ps) == 4:
                if (not 0.0 < ps[1].a < 50.0) or (not 0.0 < ps[1].e < 1.0):
                    trio_sim.remove(1)
                elif (not 0.0 < ps[2].a < 50.0) or (not 0.0 < ps[2].e < 1.0):
                    trio_sim.remove(2)
                elif (not 0.0 < ps[3].a < 50.0) or (not 0.0 < ps[3].e < 1.0):
                    trio_sim.remove(3)

            # check for merger
            if len(ps) == 4:
                if ps[1].inc == 0.0 or ps[2].inc == 0.0 or ps[3].inc == 0.0:
                    # use very small inclinations to avoid -inf
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   -3.0, -3.0, -3.0,
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
                else:
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   np.log10(ps[1].inc), np.log10(ps[2].inc), np.log10(ps[3].inc),
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
            else:
                return self.replace_trio(sim, trio_inds, trio_sim, theta1, theta2), theta1, theta2

        return np.array(states), theta1, theta2

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

    # return new sims in which planets have been merged
    def MLP_merge_planets(self, sims, trio_inds, updated_times):
        mlp_inputs = []
        thetas = []
        done_sims = []
        done_inds = []
        for i, sim in enumerate(sims):
            out, theta1, theta2 = self.generate_input(sim, trio_inds[i])

            if isinstance(out, np.ndarray):
                # if integration completed, record MLP input
                mlp_inputs.append(out)
                thetas.append([theta1, theta2])
            else:
                # if collision/merger occurred, save sim
                done_sims.append(out)
                done_inds.append(i)

        if len(mlp_inputs) > 0:
            # predict which planets collide with classification model
            class_inputs = np.array(mlp_inputs)
            class_outputs = self.class_model.class_model.make_pred(class_inputs)

            # sample predicted probabilities
            rand_nums = np.random.rand(len(class_inputs))
            col_inds = np.zeros(len(class_inputs))
            for i, rand_num in enumerate(rand_nums):
                if rand_num < class_outputs[i][0]:
                    col_inds[i] = 0
                elif rand_num < class_outputs[i][0] + class_outputs[i][1]:
                    col_inds[i] = 1
                else:
                    col_inds[i] = 2

            # re-order input array based on col_inds
            reg_inputs = []
            for i, col_ind in enumerate(col_inds):
                masses = mlp_inputs[i][:3]
                orb_elements = mlp_inputs[i][3:]

                if col_ind == 0: # collision between planets 1 and 2
                    ordered_masses = masses
                    ordered_orb_elements = orb_elements
                elif col_ind == 1: # collision between planets 2 and 3
                    ordered_masses = np.array([masses[1], masses[2], masses[0]])
                    ordered_orb_elements = np.column_stack((orb_elements[1::3], orb_elements[2::3], orb_elements[0::3])).flatten()
                elif col_ind == 2: # collision between planets 1 and 3
                    ordered_masses = np.array([masses[0], masses[2], masses[1]])
                    ordered_orb_elements = np.column_stack((orb_elements[0::3], orb_elements[2::3], orb_elements[1::3])).flatten()

                reg_inputs.append(np.concatenate((ordered_masses, ordered_orb_elements)))

            # predict orbital elements with regression model
            reg_inputs = np.array(reg_inputs)
            reg_outputs = self.reg_model.reg_model.make_pred(reg_inputs)

            m1s = 10**reg_inputs[:,0] + 10**reg_inputs[:,1] # new planet
            m2s = 10**reg_inputs[:,2] # surviving planet
            a1s = reg_outputs[:,0]
            a2s = reg_outputs[:,1]
            e1s = 10**reg_outputs[:,2]
            e2s = 10**reg_outputs[:,3]
            inc1s = 10**reg_outputs[:,4]
            inc2s = 10**reg_outputs[:,5]

        new_sims = []
        j = 0 # index for new sims array
        k = 0 # index for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                new_sims.append(done_sims[j])
                j += 1
            else:                
                # create sim that contains state of two predicted planets
                new_state_sim = rb.Simulation()
                new_state_sim.G = 4*np.pi**2 # units in which a1=1.0 and P1=1.0
                new_state_sim.add(m=1.00)
                try:
                    new_state_sim.add(m=m1s[k], a=a1s[k], e=e1s[k], inc=inc1s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                try:
                    new_state_sim.add(m=m2s[k], a=a2s[k], e=e2s[k], inc=inc2s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                new_state_sim.move_to_com()

                # replace trio with predicted duo (or single/zero if planets have unphysical orbital elements)
                new_sims.append(self.replace_trio(sims[i], trio_inds[i], new_state_sim, thetas[k][0], thetas[k][1]))
                k += 1
        
        for i, sim in enumerate(sims):
            idx = self.sims.index(sim) # find index in original list
            self.sims[idx] = new_sims[i]
            self.sims[idx].t = updated_times[i]
    
    # convert sim back to units of input sims
    def _revert_sim_units(self):
        revertedsims = []
        for j, sim in enumerate(self.sims):
            original_Mstar = self.original_Mstars[j]
            original_a1 = self.original_a1s[j]
            original_P1 = self.original_P1s[j]

            sim_copy = rb.Simulation()
            sim_copy.G = self.original_G # set G
            
            # set units
            if not (self.original_units['length'] is None or self.original_units['mass'] is None or self.original_units['time'] is None):
                sim_copy.units = self.original_units
                
            sim_copy.add(m=original_Mstar)
            ps = sim.particles
            for i in range(1, sim.N):
                sim_copy.add(m=ps[i].m*original_Mstar, a=ps[i].a*original_a1, e=ps[i].e, inc=ps[i].inc, pomega=ps[i].pomega, Omega=ps[i].Omega, theta=ps[i].theta)
                
            sim_copy.t = sim.t*original_P1
            revertedsims.append(sim_copy)
  
        return revertedsims

    def _get_orbsmax(self, tmax): # internal function with logic for initializing orbsmax as an array and checking for warnings
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
