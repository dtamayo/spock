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

# make a copy of sim that only includes the particles with inds in p_inds and (optionally) has a1 = M_star = 1.00
def get_sim_copy(sim, p_inds, set_a1_to_unity=True):
    sim_copy = rebound.Simulation()
    sim_copy.G = 4*np.pi**2 # use units in which a1=1.0, P1=1.0
    sim_copy.add(m=1.00)
    
    ps = sim.particles
    a1 = ps[int(min(p_inds))].a
    for i in range(1, sim.N):
        if i in p_inds:
            sim_copy.add(m=ps[i].m/sim.particles[0].m, a=ps[i].a/a1, e=ps[i].e, inc=ps[i].inc, pomega=ps[i].pomega, Omega=ps[i].Omega, theta=ps[i].theta)

    if not set_a1_to_unity:
        for i in range(1, sim_copy.N):
            sim_copy.particles[i].a = a1*sim_copy.particles[i].a
            
    return sim_copy

# perfect inelastic merger (taken from REBOUND)
def perfect_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents
    ps = sim.particles

    # note that p1 < p2 is not guaranteed
    i = collided_particles_index.p1
    j = collided_particles_index.p2

    total_mass = ps[i].m + ps[j].m
    merged_planet = (ps[i]*ps[i].m + ps[j]*ps[j].m)/total_mass # conservation of momentum
    merged_radius = (ps[i].r**3 + ps[j].r**3)**(1/3) # merge radius assuming a uniform density

    ps[i] = merged_planet   # update p1's state vector (mass and radius will need to be changed)
    ps[i].m = total_mass    # update to total mass
    ps[i].r = merged_radius # update to joined radius

    sim.stop() # stop sim
    return 2 # remove particle with index j

