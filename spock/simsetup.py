import numpy as np
import rebound

def collision(reb_sim, col): # for random testset run with old version of rebound that doesn't do this automatiaclly
    reb_sim.contents._status = 5
    return 0

def error_check(sim, Norbits, Nout, trio):
    i1, i2, i3 = trio
    if not isinstance(i1, int) or not isinstance(i2, int) or not isinstance(i3, int):
        raise AttributeError("SPOCK  Error: Particle indices passed to spock_features were not integers")
    ps = sim.particles
    if ps[0].m < 0 or ps[1].m < 0 or ps[2].m < 0 or ps[3].m < 0: 
        raise AttributeError("SPOCK Error: Particles in sim passed to spock_features had negative masses")

    # check periods ascending
    return

def set_timestep(sim, dtfrac):
    minTperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in sim.particles[1:sim.N_real]])
    sim.dt = dtfrac*minTperi # Wisdom 2015 suggests 0.05, 0.07 fine for short integrations

def set_sim_parameters(sim):
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

    for p in sim.particles[1:]:
        rH = p.a*(p.m/3./sim.particles[0].m)**(1./3.)
        p.r = rH

def rescale(sim):
    # returns a new simulation with particles ordered in increasing orbital period and units scaled to ps[1].a, ps[1].P, approx ps[0].m
    # also moves to COM frame
    # other parameters (like  timestep not rescaled)

    # want to add planets in order of increasing orbital period so we scale everything relative to innermost planet
    ps = [sim.particles[0]] + sorted([p for p in sim.particles[1:sim.N_real]], key=lambda p : p.P)
    simr = rebound.Simulation()

    o = ps[1].calculate_orbit(primary=ps[0]) # want 2-body orbit for innermost planet, not jacobi orbit when not ps[1]
    dscale = o.a
    tscale = o.P
    mscale = ps[0].m + ps[1].m # since above a and P solve a reduced 2 body problem with central mass M+m
    vscale = dscale/tscale

    # with above units G always = 4pi^2 since 4pi^2/P1^2 a1^3 = G(M+m1)
    simr.G = 4*np.pi**2

    for p in ps:
        simr.add(m=p.m/mscale, x=p.x/dscale, y=p.y/dscale, z=p.z/dscale,
                 vx=p.vx/vscale, vy=p.vy/vscale, vz=p.vz/vscale, r=p.r/dscale, hash=p.hash)
    
    simr.move_to_com()

    return simr
