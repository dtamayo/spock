import numpy as np
import rebound

def check_hyperbolic(sim):
    orbits = sim.orbits()
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

# function to get planet radii from their masses (according to Wolfgang+2016)
def get_rad(m):
    rad = (m/(2.7*3.0e-6))**(1/1.3)
    return rad*4.26e-4 # units of innermost a (assumed to be ~0.1AU)

# calculate the angles by which system must be rotated to have z-axis aligned with L (taken from celmech)
def _compute_transformation_angles(sim):
    Gtot_vec = np.array(sim.angular_momentum())
    Gtot = np.sqrt(Gtot_vec @ Gtot_vec)
    Ghat = Gtot_vec/Gtot
    Ghat_z = Ghat[-1]
    Ghat_perp = np.sqrt(1 - Ghat_z**2)
    theta1 = np.pi/2 - np.arctan2(Ghat[1], Ghat[0])
    theta2 = np.pi/2 - np.arctan2(Ghat_z, Ghat_perp)
    return theta1, theta2

# transform x, y, z vectors according to Euler angles Omega, I, omega (taken from celmech)
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

# align z-axis of simulation with angular momentum vector (taken from celmech)
def align_simulation(sim):
    theta1, theta2 = _compute_transformation_angles(sim)
    for p in sim.particles[:sim.N_real]:
        p.x, p.y, p.z = npEulerAnglesTransform(p.xyz, 0, theta2, theta1)
        p.vx, p.vy, p.vz = npEulerAnglesTransform(p.vxyz, 0, theta2, theta1)
    
    return theta1, theta2

# replace particle in sim with new state (in place)
def replace_p(sim, p_ind, new_particle):
    sim.particles[p_ind].m = new_particle.m
    sim.particles[p_ind].a = new_particle.a
    sim.particles[p_ind].e = new_particle.e
    sim.particles[p_ind].inc = new_particle.inc
    sim.particles[p_ind].pomega = new_particle.pomega
    sim.particles[p_ind].Omega = new_particle.Omega
    sim.particles[p_ind].l = new_particle.l
    
# return sim in which planet trio has been replaced with two planets
# with periods rescaled back to match the period of the innermost body prior in the original sim (prior to merger)
def replace_trio(original_sim, trio_inds, new_state_sim):
    sim_copy = original_sim.copy()

    new_ps = new_state_sim.particles
    original_P1 = original_sim.particles[int(trio_inds[0])].P
    for i in range(1, len(new_ps)): 
        new_ps[i].P = new_ps[i].P*original_P1

    # replace particles
    ind1, ind2, ind3 = int(trio_inds[0]), int(trio_inds[1]), int(trio_inds[2])
    if len(new_ps) == 3:
        replace_p(sim_copy, ind1, new_ps[1])
        replace_p(sim_copy, ind2, new_ps[2])
        sim_copy.remove(ind3)
    if len(new_ps) == 2:
        replace_p(sim_copy, ind1, new_ps[1])
        sim_copy.remove(ind3)
        sim_copy.remove(ind2)
    if len(new_ps) == 1:
        sim_copy.remove(ind3)
        sim_copy.remove(ind2)
        sim_copy.remove(ind1)

    # re-order particles in ascending semi-major axis
    ps = sim_copy.particles
    semi_as = []
    for i in range(1, len(ps)):
        semi_as.append(ps[i].a)
    sort_inds = np.argsort(semi_as)

    ordered_sim = sim_copy.copy()
    for i, ind in enumerate(sort_inds):
        replace_p(ordered_sim, i+1, ps[int(ind)+1])
    
    ordered_sim.original_G = original_sim.original_G
    ordered_sim.original_P1 = original_sim.original_P1
    ordered_sim.original_Mstar = original_sim.original_Mstar

    return ordered_sim

def sim_subset(sim, p_inds, copy_time=False):
    sim_copy = rebound.Simulation()
    sim_copy.G = sim.G
    ps = sim.particles
    
    if copy_time:
        sim_copy.t = sim.t
   
    sim_copy.add(m=ps[0].m)
    for i in range(1, sim.N):
        if i in p_inds:
            sim_copy.add(m=ps[i].m, a=ps[i].a, e=ps[i].e, inc=ps[i].inc, pomega=ps[i].pomega, Omega=ps[i].Omega, theta=ps[i].theta)
        
    return sim_copy

def scale_sim(sim, p_inds):
    """
    Make a copy of sim that only includes the particles with inds in p_inds 
    (assumes particles in sim are ordered from shortest to longest orbital period).
    """
    sim_copy = rebound.Simulation()
    ps = sim.particles
    
    P1 = ps[int(min(p_inds))].P
    Mstar = sim.particles[0].m
    
    sim_copy.original_G = sim.G
    sim_copy.original_P1 = P1
    sim_copy.original_Mstar = Mstar
    
    sim_copy.G = 4*np.pi**2 # use units in which a1=1.0, P1=1.0
    sim_copy.add(m=1.00)
    for i in range(1, sim.N):
        if i in p_inds:
            sim_copy.add(m=ps[i].m/Mstar, P=ps[i].P/P1, e=ps[i].e, inc=ps[i].inc, pomega=ps[i].pomega, Omega=ps[i].Omega, theta=ps[i].theta)

    sim_copy.t = sim.t/P1
    return sim_copy

def revert_sim_units(sims):
    """
    Convert sim back to units of input sims
    """
    try:
        revertedsims = []
        for i, sim in enumerate(sims):
            sim_copy = rebound.Simulation()
            sim_copy.G = sim.original_G

            sim_copy.add(m=sim.original_Mstar)
            ps = sim.particles
            for j in range(1, sim.N):
                sim_copy.add(m=ps[j].m*sim.original_Mstar, P=ps[j].P*sim.original_P1, e=ps[j].e, inc=ps[j].inc, pomega=ps[j].pomega, Omega=ps[j].Omega, theta=ps[j].theta)
            sim_copy.t = sim.t*sim.original_P1
            revertedsims.append(sim_copy)
    except AttributeError:
        raise AttributeError("sim passed to revert_units didn't have original values stored.")

    return revertedsims

def remove_ejected_ps(sims):
    new_sims = []
    for sim in sims:
        p_inds = []
        ps = sim.particles
        for i in range(1, len(ps)):
            if (0.0 < ps[i].a < 50.0) and (0.0 <= ps[i].e < 1.0):
                p_inds.append(i)
        new_sims.append(sim_subset(sim, p_inds, copy_time=True))
        
    return new_sims