import numpy as np
import os
import torch
import time
import warnings
import rebound as rb
from spock import DeepRegressor

#function to get planet radii from their masses (according to Wolfgang+2016)
def get_rad(m):
    rad = (m/(2.7*3.0e-6))**(1/1.3)
    return rad*4.26e-4 #units of innermost a (assumed to be ~0.1AU)

#calculate the angles by which system must be rotated to have z-axis aligned with L
def _compute_transformation_angles(sim):
    Gtot_vec = np.array(sim.angular_momentum())
    Gtot = np.sqrt(Gtot_vec @ Gtot_vec)
    Ghat = Gtot_vec/Gtot
    Ghat_z = Ghat[-1]
    Ghat_perp = np.sqrt(1 - Ghat_z**2)
    theta1 = np.pi/2 - np.arctan2(Ghat[1], Ghat[0])
    theta2 = np.pi/2 - np.arctan2(Ghat_z, Ghat_perp)
    return theta1, theta2

#tranform x, y, z vectors according to Euler angles Omega, I, omega
def npEulerAnglesTransform(xyz, Omega, I, omega):
    x, y, z = xyz
    s1, c1 = np.sin(omega), np.cos(omega)
    x1 = c1*x - s1*y
    y1 = s1*x + c1*y
    z1 = z

    s2, c2 = np.sin(I), np.cos(I)
    x2 = x1
    y2 = c2*y1 - s2*z1
    z2 = s2*y1 + c2*z1

    s3, c3 = np.sin(Omega), np.cos(Omega)
    x3 = c3*x2 - s3*y2
    y3 = s3*x2 + c3*y2
    z3 = z2

    return np.array([x3,y3,z3])

#align z-axis of simulation with angular momentum vector
def align_simulation(sim):
    theta1, theta2 = _compute_transformation_angles(sim)
    for p in sim.particles[:sim.N_real]:
        p.x, p.y, p.z = npEulerAnglesTransform(p.xyz, 0, theta2, theta1)
        p.vx, p.vy, p.vz = npEulerAnglesTransform(p.vxyz, 0, theta2, theta1)
    
    return theta1, theta2

#make a copy of sim that only includes the particles with inds in p_inds and has a1=p1=1.0
def get_sim_copy(sim, Npl, p_inds):
    sim_copy = rb.Simulation()
    sim_copy.G = 4*np.pi**2 #use units in which a1=1.0, P1=1.0
    sim_copy.add(m=sim.particles[0].m)
    
    ps = sim.particles
    a1 = ps[int(min(p_inds))].a
    for i in range(Npl+1):
        if i in p_inds:
            sim_copy.add(m=ps[i].m, a=ps[i].a/a1, e=ps[i].e, inc=ps[i].inc, pomega=ps[i].pomega, Omega=ps[i].Omega, theta=ps[i].theta)

    return sim_copy

#perfect inelastic merger
def perfect_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents
    ps = sim.particles

    #note that p1 < p2 is not guaranteed
    i = collided_particles_index.p1
    j = collided_particles_index.p2

    total_mass = ps[i].m + ps[j].m
    merged_planet = (ps[i]*ps[i].m + ps[j]*ps[j].m)/total_mass #conservation of momentum
    merged_radius = (ps[i].r**3 + ps[j].r**3)**(1/3) #merge radius assuming a uniform density

    ps[i] = merged_planet   #update p1's state vector (mass and radius will need to be changed)
    ps[i].m = total_mass    #update to total mass
    ps[i].r = merged_radius #update to joined radius

    sim.stop() #stop sim
    return 2 #remove particle with index j

#planet formation simulation model
class GiantImpactPhaseEmulator():
    #initialize function
    def __init__(self, sims, reg_model_file='collision_orbital_outcome_regressor.torch', class_model_file='collision_merger_classifier.torch'):
        #load regression and classification models
        pwd = os.path.dirname(__file__)
        self.reg_model = torch.load(pwd + '/models/' + reg_model_file, map_location=torch.device('cpu'))
        self.class_model = torch.load(pwd + '/models/' + class_model_file, map_location=torch.device('cpu'))

        #load SPOCK model
        self.deep_model = DeepRegressor()

        try:
            self.sims = [self.run_short_sim(sim) for sim in sims]
        except:
            self.sims = [self.run_short_sim(sims)] # single simulation

    #replace particle in sim with new state (in place)
    def replace_p(self, sim, p_ind, new_particle):
        sim.particles[p_ind].m = new_particle.m
        sim.particles[p_ind].a = new_particle.a
        sim.particles[p_ind].e = new_particle.e
        sim.particles[p_ind].inc = new_particle.inc
        sim.particles[p_ind].pomega = new_particle.pomega
        sim.particles[p_ind].Omega = new_particle.Omega
        sim.particles[p_ind].l = new_particle.l
    
    #return sim in which planet trio has been replaced with two planets
    def replace_trio(self, original_sim, trio_inds, new_state_sim, theta1, theta2):
        #change back to units of original sim
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
            sim_copy.remove(ind2) #does this work?
        if len(new_ps) == 1:
            sim_copy = original_sim.copy()
            sim_copy.remove(ind3)
            sim_copy.remove(ind2) #does this work?
            sim_copy.remove(ind1) #does this work?
        
        #change axis orientation back to original sim here
        for p in sim_copy.particles[:sim_copy.N_real]:
            p.x, p.y, p.z = npEulerAnglesTransform(p.xyz, -theta1, -theta2, 0)
            p.vx, p.vy, p.vz = npEulerAnglesTransform(p.vxyz, -theta1, -theta2, 0)
        
        #re-order particles in ascending semi-major axis
        ps = sim_copy.particles
        semi_as = []
        for i in range(1, len(ps)):
            semi_as.append(ps[i].a)
        sort_inds = np.argsort(semi_as)

        ordered_sim = sim_copy.copy()
        for i in range(len(sort_inds)):
            self.replace_p(ordered_sim, i+1, ps[int(sort_inds[i])+1])

        return ordered_sim

    #run short sim to get input for MLP model (returns a sim if merger/ejection occurs)
    def generate_input(self, sim, trio_inds):
        #get three-planet sim
        trio_sim = get_sim_copy(sim, len(sim.particles)-1, trio_inds)
        ps = trio_sim.particles
            
        #align z-axis with direction of angular momentum
        theta1, theta2 = align_simulation(trio_sim)

        #assign planet radii
        for i in range(1, len(ps)):
            ps[i].r = get_rad(ps[i].m)

        #set integration settings
        trio_sim.integrator = 'mercurius'
        trio_sim.collision = 'direct'
        trio_sim.collision_resolve = perfect_merge
        Ps = np.array([p.P for p in ps[1:len(ps)]])
        es = np.array([p.e for p in ps[1:len(ps)]])
        minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
        trio_sim.dt = 0.05*minTperi

        times = np.linspace(0.0, 1e4, 100)
        states = [np.log10(ps[1].m), np.log10(ps[2].m), np.log10(ps[3].m)]
        for i in range(len(times)):
            trio_sim.integrate(times[i], exact_finish_time=0)

            #check for ejected planets
            if len(ps) == 4:
                if (not 0.0 < ps[1].a < 50.0) or (not 0.0 < ps[1].e < 1.0):
                    trio_sim.remove(1)
                elif (not 0.0 < ps[2].a < 50.0) or (not 0.0 < ps[2].e < 1.0):
                    trio_sim.remove(2)
                elif (not 0.0 < ps[3].a < 50.0) or (not 0.0 < ps[3].e < 1.0):
                    trio_sim.remove(3)

            #check for merger
            if len(ps) == 4:
                if ps[1].inc == 0.0 or ps[2].inc == 0.0 or ps[3].inc == 0.0:
                    #use very small inclinations to avoid -inf
                    states.extend([ps[1].a, ps[2].a, ps[3].a,
                                   np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                   -3.0, -3.0, -3.0,
                                   np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                   np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                   np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                   np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])
                else:
                    #save state
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

    #get unstable trios for list of sims using SPOCK deep model
    def get_unstable_trios(self, sims):
        trio_sims = []
        trio_inds = []
        Npls = [sim.N-1 for sim in sims]
        
        for i in range(len(sims)):
            for j in range(Npls[i] - 2):
                trio_inds.append([j+1, j+2, j+3])
                trio_sims.append(get_sim_copy(sims[i], Npls[i], [j+1, j+2, j+3]))

        t_insts, _, _ = self.deep_model.predict_instability_time(trio_sims, samples=100, max_model_samples=10, Ncpus=8)

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

    #return new sims in which planets have been merged
    def MLP_merge_planets(self, sims, trio_inds, times_after_merge):
        mlp_inputs = []
        thetas = []
        done_sims = []
        done_inds = []
        for i in range(len(sims)):
            out, theta1, theta2 = self.generate_input(sims[i], trio_inds[i])

            if isinstance(out, np.ndarray):
                #if integration completed, record MLP input
                mlp_inputs.append(out)
                thetas.append([theta1, theta2])
            else:
                #if collision/merger occurred, save sim
                done_sims.append(out)
                done_inds.append(i)

        if len(mlp_inputs) > 0:
            #predict which planets collide with classification model
            class_inputs = np.array(mlp_inputs)
            class_outputs = self.class_model.make_pred(class_inputs)

            #sample predicted probabilities
            rand_nums = np.random.rand(len(class_inputs))
            col_inds = np.zeros(len(class_inputs))
            for i in range(len(rand_nums)):
                if rand_nums[i] < class_outputs[i][0]:
                    col_inds[i] = 0
                elif rand_nums[i] < class_outputs[i][0] + class_outputs[i][1]:
                    col_inds[i] = 1
                else:
                    col_inds[i] = 2

            #re-order input array based on col_inds
            reg_inputs = []
            for i in range(len(col_inds)):
                masses = mlp_inputs[i][:3]
                orb_elements = mlp_inputs[i][3:]

                if col_inds[i] == 0: #collision between planets 1 and 2
                    ordered_masses = masses
                    ordered_orb_elements = orb_elements
                elif col_inds[i] == 1: #collision between planets 2 and 3
                    ordered_masses = np.array([masses[1], masses[2], masses[0]])
                    ordered_orb_elements = np.column_stack((orb_elements[1::3], orb_elements[2::3], orb_elements[0::3])).flatten()
                elif col_inds[i] == 2: #collision between planets 1 and 3
                    ordered_masses = np.array([masses[0], masses[2], masses[1]])
                    ordered_orb_elements = np.column_stack((orb_elements[0::3], orb_elements[2::3], orb_elements[1::3])).flatten()

                reg_inputs.append(np.concatenate((ordered_masses, ordered_orb_elements)))

            #predict orbital elements with regression model
            reg_inputs = np.array(reg_inputs)
            reg_outputs = self.reg_model.make_pred(reg_inputs)

            m1s = 10**reg_inputs[:,0] + 10**reg_inputs[:,1] #new planet
            m2s = 10**reg_inputs[:,2] #surviving planet
            a1s = reg_outputs[:,0]
            a2s = reg_outputs[:,1]
            e1s = 10**reg_outputs[:,2]
            e2s = 10**reg_outputs[:,3]
            inc1s = 10**reg_outputs[:,4]
            inc2s = 10**reg_outputs[:,5]

        new_sims = []
        j = 0 #ind for new sims array
        k = 0 #ind for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                new_sims.append(done_sims[j])
                j += 1
            else:                
                #create sim that contains state of two predicted planets
                new_state_sim = rb.Simulation()
                new_state_sim.G = 4*np.pi**2 #units in which a1=1.0 and P1=1.0 (initially)
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

                #replace trio with predicted duo (or single/zero if planets have unphysical orbital elements)
                new_sims.append(self.replace_trio(sims[i], trio_inds[i], new_state_sim, thetas[k][0], thetas[k][1]))
                k += 1
        
        for i, sim in enumerate(sims):
            idx = self.sims.index(sim) # find index in original list
            self.sims[idx] = new_sims[i]
            self.sims[idx].t = times_after_merge[i]

    def run_short_sim(self, original_sim):
        Npl = len(original_sim.particles) - 1
        original_a1 = original_sim.particles[1].a
        sim = get_sim_copy(original_sim, Npl, np.arange(1, Npl+1))

        #assign planet radii
        ps = sim.particles
        for i in range(1, len(ps)):
            ps[i].r = get_rad(ps[i].m)

        #set integration settings
        sim.integrator = 'mercurius'
        sim.collision = 'direct'
        sim.collision_resolve = 'merge'
        Ps = np.array([p.P for p in ps[1:len(ps)]])
        es = np.array([p.e for p in ps[1:len(ps)]])
        minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
        sim.dt = 0.05*minTperi

        #integrate for 1e4 orbits
        sim.integrate(1e4)

        #remove particle radii
        for i in range(1, len(ps)):
            ps[i].r = 0.0

        #re-scale lengths
        for i in range(1, len(ps)):
            ps[i].a = ps[i].a*original_a1

        #reset to defaults
        sim.integrator = 'ias15'
        sim.dt = 0.001
        sim.collision = 'none'
        sim.t = 0.0

        return sim
    
    def integrate(self, tmax=1e9):
        while np.min([sim.t for sim in self.sims]) < tmax: # take another step if any sims are still at t<tmax
            self.step(tmax)
    
    def step(self, tmax):
        for sim in self.sims: # assume all 2 planet systems (N=3) are stable (could modify to Hill stability criterion)
            if sim.N < 4:
                sim.t = tmax
        
        sims_to_update = [sim for sim in self.sims if sim.t < tmax]
        #estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            return
    
        t_insts, trio_inds = self.get_unstable_trios(sims_to_update)
  
        #get list of sims for which planets need to be merged
        sims_to_merge = []
        trios_to_merge = []
        times_after_merge = []
        for i, sim in enumerate(sims_to_update):
            idx = self.sims.index(sim)      # get index in original list
            time_after_merge = sim.t + t_insts[i]
            if time_after_merge > tmax:   # won't merge before tmax, so update to tmax
                self.sims[idx].t = tmax
            else:
                sims_to_merge.append(sim)
                trios_to_merge.append(trio_inds[i])
                times_after_merge.append(time_after_merge)
                
        #get new sims with planets merged
        self.MLP_merge_planets(sims_to_merge, trios_to_merge, times_after_merge)
