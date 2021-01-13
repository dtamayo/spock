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
    assert isinstance(sim, rebound.Simulation)
    ps = sim.particles
    ms = np.array([p.m for p in sim.particles[:sim.N_real]])
    if np.min(ms) < 0: # at least one body has a mass < 0
        raise AttributeError("SPOCK Error: Particles in sim passed to spock_features had negative masses")

    if np.max(ms) != ms[0]:
        raise AttributeError("SPOCK Error: Particle at index 0 must be the primary (dominant mass)")

    return

def set_integrator_and_timestep(sim):
    Ps = np.array([p.P for p in sim.particles[1:sim.N_real]])
    es = np.array([p.e for p in sim.particles[1:sim.N_real]])
    if np.max(es) < 1:
        minTperi = np.min(Ps*(1-es)**1.5/np.sqrt(1+es)) # min peri passage time
        sim.dt = 0.05*minTperi                          # Wisdom 2015 suggests 0.05
    else:                                               # hyperbolic orbit 
        sim.dt = np.nan # so tseries gives nans, but still always gives same shape array

    if np.max(es) > 0.99:                               # avoid stall with WHFAST for e~1
        sim.integrator = "ias15"
    else:
        sim.integrator = "whfast"

def init_sim_parameters(sim, megno=True, safe_mode=1): 
    # if megno=False and safe_mode=0, integration will be 2x faster. But means we won't get the same trajectory realization for the systems in the training set, but rather a different equally valid realization. We've tested that this doesn't affect the performance of the model (as it shouldn't!).

    check_valid_sim(sim)

    try:
        sim.collision = 'line'  # use line if using newer version of REBOUND
    except:
        sim.collision = 'direct'# fall back for older versions

    maxd = np.array([p.d for p in sim.particles[1:sim.N_real]]).max()
    sim.exit_max_distance = 100*maxd
                
    sim.ri_whfast.keep_unsynchronized = 0
    sim.ri_whfast.safe_mode = safe_mode 

    if sim.N_var == 0 and megno: # no variational particles
        sim.init_megno(seed=0)
   
    set_integrator_and_timestep(sim)

    # Set particle radii to their individual Hill radii. 
    # Exact collision condition doesn't matter, but this behaves at extremes.
    # Imagine huge M1, tiny M2 and M3. Don't want to set middle planet's Hill 
    # sphere to mutual hill radius with huge M1 when catching collisions w/ M3
    
    for p in sim.particles[1:sim.N_real]:
        rH = p.a*(p.m/3./sim.particles[0].m)**(1./3.)
        p.r = rH
    
    sim.move_to_com()
