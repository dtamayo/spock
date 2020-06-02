import numpy as np
import rebound

def check_hyperbolic(sim):
    orbits = sim.calculate_orbits()
    amin = np.min([o.a for o in orbits])
    if amin < 0: # at least one orbit is hyperbolic (a<0)
        return True
    else:
        return False

def check_valid_sim(sim):
    ps = sim.particles
    ms = np.array([p.m for p in sim.particles[:sim.N_real]])
    if np.min(ms) < 0: # at least one body has a mass < 0
        raise AttributeError("SPOCK Error: Particles in sim passed to spock_features had negative masses")

    if np.max(ms) != ms[0]:
        raise AttributeError("SPOCK Error: Particle at index 0 must be the primary (dominant mass)")

    return

def set_integrator_and_timestep(sim, dtfrac):
    Ps = np.array([p.P for p in sim.particles[1:sim.N_real]])
    es = np.array([p.e for p in sim.particles[1:sim.N_real]])
    minTperi = np.min(Ps * (1-es)**1.5 / np.sqrt(1+es)) # minimum time of pericenter passages

    sim.dt = dtfrac*minTperi # Wisdom 2015 suggests 0.05


    if np.max(es) > 0.99: # approx threshold where ias15 becomes faster than whfast
        sim.integrator = "ias15"
    else:
        sim.integrator = "whfast"

def collision(reb_sim, col): # for random testset run with old version of rebound that doesn't do this automatiaclly
    reb_sim.contents._status = 5
    return 0

def init_sim_parameters(sim, dtfrac=0.05): 
    # overwrites particle radii with their individual Hill radii
    # sets up collision detection (will raise rebound.Escape exception with new versions of REBOUND, will set sim._status = 5 for all versions)
    # sets safe_mode to 1 for consistent MEGNO calculation
    # sets integrator to whfast unless ecc > 0.99 (then IAS)
    
    check_valid_sim(sim)

    try:
        sim.collision = 'line'  # use line if using newer version of REBOUND
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
   
    set_integrator_and_timestep(sim, dtfrac=dtfrac)

    # Set particle radii to their individual Hill radii. 
    # Exact collision condition doesn't matter, but this behaves at extremes.
    # Imagine huge M1, tiny M2 and M3. Don't want to set middle planet's Hill 
    # sphere to mutual hill radius with huge M1 when catching collisions w/ M3
    
    for p in sim.particles[1:]:
        rH = p.a*(p.m/3./sim.particles[0].m)**(1./3.)
        p.r = rH
    
    sim.move_to_com()
