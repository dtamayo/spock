# Run script with python run.py sim_id, where sim_id is an integer to use for the sim_id (and the RNG seed). Runs a base system and a shadow system
import numpy as np
import rebound
from random import random, uniform, seed
import time
import sys
import math
import random
from celmech import Andoyer, Poincare, AndoyerHamiltonian
from celmech.andoyer import get_Xstarres
from scipy.integrate import ode

def logunif(r, mini, maxi):
    logmin = np.log10(mini)
    logmax = np.log10(maxi)
    return 10**(r.uniform(logmin, logmax))

def get_resonant(seed, Nplanets=3):
    r = random.Random()
    r.seed(seed)

    Phiprimecrits = [0, 1., -2./3.]
    pairs = ['inner', 'outer', 'split']
    
    k=r.randint(1,2) 
    pairindex=r.randint(0,2)
    pair = pairs[pairindex]
    m1 = logunif(r, 1.e-7, 1.e-4)
    m2 = logunif(r, 1.e-7, 1.e-4)
    eH = ((m1+m2)/3.)**(1./3.)
    ehillstable = 3.5*eH
    jmax = k/(1-1/(1+3.5*eH)**1.5)
    if pair == 'split':
        if Nplanets == 2:
            return # don't want 2planet systems 60 hill radii apart
        maxHillradii = 60. # 3rd planet will go in middle so draw up to 60
    else:
        maxHillradii = 30.
    jmin = max(k+1, k/(1-1/(1+maxHillradii*eH)**1.5))
    jmin = int(np.ceil(jmin))
    jmax = int(np.floor(jmax))
    if k == 2: # if k = 2, want odd j so we don't get e.g. 8:6 = 4:3
        if jmin % 2 == 0: # even
            jmin += 1
        if jmax % 2 == 0:
            jmax -= 1
    j = r.randrange(jmin, jmax+1, k) # choose randomly between limits in steps of k e.g. (3,5,7,9)
    a1 = 1.
    a2 = (float(j)/(j-k))**(2./3.)
    ecross1 = (a2-a1)/a1
    ecross2 = (a2-a1)/a2
    emin1 = m2/ecross1**2
    emin2 = m1/ecross2**2
    emin = max(emin1, emin2) # take as min Z the larger of the kicks a planet gets at conjunction       
    emin = max(emin, (m1+m2)**(1./k)) # below mtot^1/k, the resonant term is smaller than the second order mass terms we ignore
    emax = min(ecross1, ecross2)
    
    avars = Andoyer(j=j,k=k,X=0,Y=0,m1=m1, m2=m2)
    Phiprimecrit = Phiprimecrits[k]
    Xcrit = get_Xstarres(k, Phiprimecrit)
    Phicrit = 0.5*Xcrit**2
    emin = max(avars.Phi_to_Z(Phicrit), emin) # first quantity is value of Z at bifurcation when res first appears
    
    Zstar=logunif(r, emin, emax)
    libfac=logunif(r, 3.e-3, 3)
    negative = r.randint(0,1)
    if negative:
        libfac *= -1
   
    Zcom = logunif(r, emin, emax)
    avars = Andoyer.from_elements(j=j,k=k,Zstar=Zstar,libfac=libfac, m1=m1, m2=m2, Zcom=Zcom, phiZcom=r.uniform(0,2*np.pi), theta=r.uniform(0,2*np.pi), theta1=r.uniform(0,2*np.pi))
    tmax = r.uniform(0,10*avars.tlib)
    H = AndoyerHamiltonian(avars)
    H.integrate(tmax)
    pvars = avars.to_Poincare()
    ps = pvars.particles
    if Nplanets==3:
        m3 = logunif(r, 1.e-7, 1.e-4)
        pvarssorted = Poincare(G=pvars.G)

        if pair == "inner":
            eH = ((m2+m3)/3.)**(1./3.)
            beta = r.uniform(3.5,30)
            a3 = a2*(1+beta*eH)
            ecross3 = (a3-a2)/a3
            emin3 = m2/ecross3**2
            e3 = logunif(r, emin3, ecross3)
            pvarssorted.add(m=ps[1].m, M=ps[1].M, a=ps[1].a, e=ps[1].e, gamma=ps[1].gamma, l=ps[1].l)
            pvarssorted.add(m=ps[2].m, M=ps[2].M, a=ps[2].a, e=ps[2].e, gamma=ps[2].gamma, l=ps[2].l)
            pvarssorted.add(m=m3, M=1, a=a3, e=e3, gamma=r.uniform(0, 2*np.pi), l=r.uniform(0, 2*np.pi))

        elif pair == "outer":
            eH = ((m1+m3)/3.)**(1./3.)
            beta = r.uniform(3.5,30)
            a3 = a1/(1+beta*eH)
            ecross3 = (a1-a3)/a3
            emin3 = m1/ecross3**2
            e3 = logunif(r, emin3, ecross3)
            pvarssorted.add(m=m3, M=1, a=a3, e=e3, gamma=r.uniform(0, 2*np.pi), l=r.uniform(0, 2*np.pi))
            pvarssorted.add(m=ps[1].m, M=ps[1].M, a=ps[1].a, e=ps[1].e, gamma=ps[1].gamma, l=ps[1].l)
            pvarssorted.add(m=ps[2].m, M=ps[2].M, a=ps[2].a, e=ps[2].e, gamma=ps[2].gamma, l=ps[2].l)

        elif pair == "split":
            eH1 = ((m1+m3)/3.)**(1./3.)
            eH2 = ((m2+m3)/3.)**(1./3.)
            amin = a1*(1+3.5*eH1)
            amax = min(a1*(1+30*eH1), a2/(1+3.5*eH2))
            if amin > amax:
                seed += 900000
                return get_resonant(seed=seed) # draw new sample (may not be 'split')
            a3 = r.uniform(amin, amax)
            ecross3 = min((a3-a1)/a1, (a2-a3)/a3)
            emin3 = max(m1/((a3-a1)/a3)**2, m2/((a2-a3)/a3)**2)
            e3 = logunif(r, emin3, ecross3)
            pvarssorted.add(m=ps[1].m, M=ps[1].M, a=ps[1].a, e=ps[1].e, gamma=ps[1].gamma, l=ps[1].l)
            pvarssorted.add(m=m3, M=1, a=a3, e=e3, gamma=r.uniform(0, 2*np.pi), l=r.uniform(0, 2*np.pi))
            pvarssorted.add(m=ps[2].m, M=ps[2].M, a=ps[2].a, e=ps[2].e, gamma=ps[2].gamma, l=ps[2].l)
        
        sim = pvarssorted.to_Simulation()
    else:
        sim = avars.to_Simulation()
    # add inclinations and scale s.t. a1 = 1, Mprimary=1. and G = 4*pi**2
    ps = sim.particles
    
    siminc = rebound.Simulation()
    siminc.G = 4*np.pi**2
    
    dscale = ps[1].a
    tscale = ps[1].P
    mscale = siminc.G*dscale**3/tscale**2 # ps[0].m + ps[1].m
    
    siminc.add(m=ps[0].m/mscale, x=ps[0].x/dscale, y=ps[0].y/dscale, vx=ps[0].vx/dscale*tscale, vy=ps[0].vy/dscale*tscale)
    for p in ps[1:]:
        siminc.add(m=p.m/mscale, a=p.a/dscale, e=p.e, inc=logunif(r, 1.e-3, 1.e-1), Omega=r.uniform(0,2*np.pi), pomega=p.pomega, l=p.l)
        rH = siminc.particles[-1].a*(siminc.particles[-1].m/3./siminc.particles[0].m)**(1./3.) 
        siminc.particles[-1].r = rH
    siminc.move_to_com()
    siminc.integrator="whfast"
    siminc.dt = 2.*np.sqrt(3)/100.*siminc.particles[1].P
    siminc.ri_whfast.safe_mode=0
    siminc.collision="line"
    return siminc, j, k, pairindex, Zstar, libfac, Zcom

def run_resonant(seed, runstr, Nplanets=3, maxorbs=1.e9, shadow=False):
    originalseed = seed
    success=False
    while not success:
        try:
            sim, j, k, pairindex, Zstar, libfac, Zcom = get_resonant(seed, Nplanets)
            sim.integrate(1.e4*sim.particles[1].P)
            success=True
        except rebound.Collision:
            seed += 700000 # roughly relatively prime with amin > amax offset above
    sim, j, k, pairindex, Zstar, libfac, Zcom = get_resonant(seed, Nplanets) # get same sim
    print('{0}, {1}, {2}, {3}, {4}, {5:8e}, {6:8e}, {7:8e}\n'.format(originalseed, seed, pairindex, j, k, Zstar, libfac, Zcom))
    if shadow:
        shadowstr = 'shadow'
        kicksize=1.e-11
        sim.particles[2].x += kicksize
    else:
        shadowstr = ''
    filename = '../data/resonant/simulation_archives/'+shadowstr+'runs/sa'+runstr
    sim.automateSimulationArchive(filename, interval=maxorbs/1000., deletefile=True)
    try:
        sim.integrate(maxorbs*sim.particles[1].P)
    except rebound.Collision:
        sim.simulationarchive_snapshot(filename)  # save final snapshot if collision occurs

def run_resonant_condition(seed, runstr, conditionfunc, Nplanets=3, maxorbs=1.e9, shadow=False):
    originalseed = seed
    sim, j, k, pairindex, Zstar, libfac, Zcom = get_resonant(seed, Nplanets)
    if conditionfunc(sim):
        if shadow:
            shadowstr = 'shadow'
            kicksize=1.e-11
            sim.particles[2].x += kicksize
        else:
            shadowstr = ''
        filename = '../data/resonant/simulation_archives/'+shadowstr+'runs/sa'+runstr
        sim.automateSimulationArchive(filename, interval=maxorbs/1000., deletefile=True)
        try:
            sim.integrate(maxorbs*sim.particles[1].P)
        except rebound.Collision:
            sim.simulationarchive_snapshot(filename)  # save final snapshot if collision occurs

def collision(reb_sim, col):
    reb_sim.contents._status = 5 # causes simulation to stop running and have flag for whether sim stopped due to collision
    return 0

def run_random(sim_id, integrator="whfast", dt=None, maxorbs=1.e9, betamin=1., betamax=30., shadow=False, runstr=None):
    if dt is None:
        dt = 2.*math.sqrt(3)/100. # dt approx 3.5 % of innermost orbital period

    datapath = '../data/random/'

    a1 = 1. # All distances in units of the innermost semimajor axis (always at 1)
    Mstar = 1. # All masses in units of stellar mass

    logMmin = np.log10(1.e-7) # 1/3 Mars around Sun
    logMmax = np.log10(1.e-4) # 2 Nep around Sun
    logincmin = np.log10(1.e-3)
    logincmax = np.log10(1.e-1) # max mutual inclination of 11.4 degrees. Absolute of 5.7 deg

    seed(sim_id)

    M1 = 10.**uniform(logMmin, logMmax)
    M2 = 10.**uniform(logMmin, logMmax)
    M3 = 10.**uniform(logMmin, logMmax)

    hill12 = a1*((M1+M2)/3.)**(1./3.)
    beta1 = uniform(betamin, betamax)
    a2 = a1 + beta1*hill12

    hill23 = a2*((M2+M3)/3.)**(1./3.)
    beta2 = uniform(betamin, betamax)
    a3 = a2 + beta2*hill23

    minhill = min(hill12, hill23)

    ecrit1 = (a2-a1)/a1
    ecrit21 = (a2-a1)/a2
    ecrit23 = (a3-a2)/a2
    ecrit3 = (a3-a2)/a3
    
    logemax1 = np.log10(ecrit1)
    logemax2 = np.log10(min(ecrit21, ecrit23))
    logemax3 = np.log10(ecrit3)

    logemin1 = np.log10(M2/ecrit1**2)
    logemin2 = np.log10(max(M1/ecrit21**2, M3/ecrit23**2))
    logemin3 = np.log10(M2/ecrit3**2)

    #print("M1 = {0}, M2 = {1}, M3 = {2}".format(M1, M2, M3))
    #print("beta1 = {0}, beta2 = {1}".format(beta1, beta2))
    #print("a1 = {0}, a2 = {1}, a3 = {2}".format(a1, a2, a3))
    #print("emax1 = {0}, emax2 = {1}, emax3 = {2}".format(10**logemax1, 10**logemax2, 10**logemax3))
    #print("emin1 = {0}, emin2 = {1}, emin3 = {2}".format(10**logemin1, 10**logemin2, 10**logemin3))
    #print("minhill = {0}".format(minhill))

    emax = 0.999
    e1 = min(10.**uniform(logemin1, logemax1), emax) # make sure ecc < 1
    e2 = min(10.**uniform(logemin2, logemax2), emax)
    e3 = min(10.**uniform(logemin3, logemax3), emax)
    
    i1 = 10.**uniform(logincmin, logincmax)
    i2 = 10.**uniform(logincmin, logincmax)
    i3 = 10.**uniform(logincmin, logincmax)

    sim = rebound.Simulation()
    sim.integrator=integrator
    sim.ri_whfast.safe_mode = 0
    sim.G = 4*np.pi**2

    sim.add(m=1.)
    sim.add(m=M1, a=a1, e=e1, pomega=random()*2.*np.pi, inc=i1, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.add(m=M2, a=a2, e=e2, pomega=random()*2.*np.pi, inc=i2, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.add(m=M3, a=a3, e=e3, pomega=random()*2.*np.pi, inc=i3, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.move_to_com()
    ps = sim.particles

    if shadow:
        kicksize=1.e-11
        ps[2].x += kicksize

    if integrator=="whfast":
        sim.dt = dt*sim.particles[1].P
    sim.collision = "direct"
    sim.collision_resolve = collision
        
    #runstr = "{0:0=7d}.bin".format(sim_id)
    if shadow:
        shadowstr = 'shadow'
    else:
        shadowstr = ''

    if runstr:
        sim.save(datapath+'initial_conditions/'+shadowstr+'runs/ic'+runstr)
        sim.initSimulationArchive(datapath+'simulation_archives/'+shadowstr+'runs/sa'+runstr, interval=maxorbs/1000.)

    E0 = sim.calculate_energy()
    t0 = time.time()
    sim.integrate(maxorbs) # will stop if collision occurs
    Ef = sim.calculate_energy()    
    Eerr = abs((Ef-E0)/E0)

    if runstr:
        sim.save(datapath+'final_conditions/'+shadowstr+'runs/fc'+runstr)

    return (sim.t, Eerr, time.time()-t0)
