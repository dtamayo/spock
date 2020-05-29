import numpy as np
import rebound

def collision(reb_sim, col): # for random testset run with old version of rebound that doesn't do this automatiaclly
    reb_sim.contents._status = 5
    return 0

def check_valid_sim(sim):
    ps = sim.particles
    if ps[0].m < 0 or ps[1].m < 0 or ps[2].m < 0 or ps[3].m < 0: 
        raise AttributeError("SPOCK Error: Particles in sim passed to spock_features had negative masses")

    # check periods ascending
    return

def set_timestep(sim, dtfrac):
    minTperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in sim.particles[1:sim.N_real]])
    sim.dt = dtfrac*minTperi # Wisdom 2015 suggests 0.05

def init_sim_parameters(sim, dtfrac=0.05): 
    # Assumes a valid sim (that passes check_valid_sim)
    # overwrites particle radii with their individual Hill radii
    # sets up collision detection (will raise Exception with new versions of REBOUND, will set sim._status = 5 for all versions)
    # sets safe_mode to 1 for consistent MEGNO calculation
    # sets integrator to whfast
    try:
        sim.collision = 'line' # use line if using newer version of REBOUND
    except:
        sim.collision = 'direct'# fall back for older versions
    
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 0
    sim.ri_whfast.safe_mode = 1

    if sim.N_var == 0: # no variational particles
        try:
            sim.init_megno(seed=0)
        except: # fallbackf or old versions of REBOUND
            sim.init_megno()
   
    sim.integrator = "whfast"
    set_timestep(sim, dtfrac=dtfrac)

    for p in sim.particles[1:]:
        rH = p.a*(p.m/3./sim.particles[0].m)**(1./3.)
        p.r = rH
    
    sim.move_to_com()

def check_hyperbolic(sim):
    orbits = sim.calculate_orbits()
    amin = np.min([o.a for o in orbits])
    if amin < 0: # at least one orbit is hyperbolic (a<0)
        return True
    else:
        return False
