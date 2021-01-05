import rebound
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict
from celmech.poincare import Poincare, PoincareHamiltonian
from celmech import Andoyer, AndoyerHamiltonian
from celmech.resonances import resonant_period_ratios, resonance_intersections_list, resonance_pratio_span
from celmech.transformations import masses_to_jacobi
from celmech.andoyer import get_num_fixed_points
import itertools

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0


def safe_run_func(runfunc):
    def new_run_func(*args, **kwargs):
        try:
            return runfunc(*args, **kwargs)
        except RuntimeError:
            return None
    return new_run_func

from scipy.optimize import brenth
def F(e,alpha,gamma):
    """Equation 35 of Laskar & Petit (2017)"""
    denom = np.sqrt(alpha*(1-e*e)+gamma*gamma*e*e)
    return alpha*e -1 + alpha + gamma*e / denom

### start AMD functions

def critical_relative_AMD(alpha,gamma):
    """Equation 29"""
    e0 = np.min((1,1/alpha-1))
    ec = brenth(F,0,e0,args=(alpha,gamma))
    e1c = np.sin(np.arctan(gamma*ec / np.sqrt(alpha*(1-ec*ec))))
    curlyC = gamma*np.sqrt(alpha) * (1-np.sqrt(1-ec*ec)) + (1 - np.sqrt(1-e1c*e1c))
    return curlyC

@safe_run_func
def compute_AMD(sim):
    pstar = sim.particles[0]
    Ltot = pstar.m * np.cross(pstar.xyz,pstar.vxyz)
    ps = sim.particles[1:]
    Lmbda=np.zeros(len(ps))
    G = np.zeros(len(ps))
    Lhat = np.zeros((len(ps),3))
    for k,p in enumerate(sim.particles[1:]):
        orb = p.calculate_orbit(primary=pstar)
        Lmbda[k] = p.m * np.sqrt(p.a)
        G[k] = Lmbda[k] * np.sqrt(1-p.e*p.e)
        hvec = np.cross(p.xyz,p.vxyz)
        Lhat[k] = hvec / np.linalg.norm(hvec)
        Ltot = Ltot + p.m * hvec
    cosi = np.array([Lh.dot(Ltot) for Lh in Lhat]) / np.linalg.norm(Ltot)
    return np.sum(Lmbda) - np.sum(G * cosi)

@safe_run_func
def AMD_stable_Q(sim):
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        if C>Ccrit:
            return False
    return True

@safe_run_func
def AMD_stability_coefficients(sim):
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    coeffs = np.zeros(len(ps)-1)
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        coeffs[i] = C / Ccrit
    return coeffs

def AMD_stability_coefficient(sim, i1, i2):
    AMD = compute_AMD(sim)
    ps = sim.particles
    pstar = ps[0]
    
    pIn = ps[i1]
    pOut = ps[i2]
    orbIn = pIn.calculate_orbit(pstar)
    orbOut = pOut.calculate_orbit(pstar)
    alpha = orbIn.a / orbOut.a
    gamma = pIn.m / pOut.m
    LmbdaOut = pOut.m * np.sqrt(orbOut.a)
    Ccrit = critical_relative_AMD(alpha,gamma)
    C = AMD / LmbdaOut
    return C / Ccrit

### end AMD functions
# write functions to take args and unpack them at top so it's clear what you have to pass in args
@safe_run_func
def orbtseries(sim, args, trio):
    Norbits = args[0]
    Nout = args[1]
    val = np.zeros((Nout, 19))

    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    ###############################
    # Chunk above should be the same in all runfuncs we write in order to match simarchives
    # Fill in values below
    
    times = np.linspace(0, Norbits*sim.particles[1].P, Nout) # TTV systems don't have ps[1].P=1, so must multiply!
   
    P0 = sim.particles[1].P
    a0 = sim.particles[1].a
    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except:
            break

        orbits = sim.calculate_orbits()
        skipped = 0
        for j, o in enumerate(orbits):
            #print(j, trio)
            if j+1 not in trio:
                skipped += 1
                continue
            #print(j, 'actually in', trio, skipped)
            val[i,0] = sim.t/P0
            val[i,6*(j-skipped)+1] = o.a/a0
            val[i,6*(j-skipped)+2] = o.e
            val[i,6*(j-skipped)+3] = o.inc
            val[i,6*(j-skipped)+4] = o.Omega
            val[i,6*(j-skipped)+5] = o.pomega
            val[i,6*(j-skipped)+6] = o.M

    return val

@safe_run_func
def orbsummaryfeaturesxgb(sim, args):
    Norbits = args[0]
    Nout = args[1]
    window = args[2]

    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    ##############################
    
    times = np.linspace(0, Norbits*sim.particles[1].P, Nout) # TTV systems don't have ps[1].P=1, so must multiply!

    ps = sim.particles
    P0 = ps[1].P
    Nout = len(times)
        
    features = OrderedDict()
    AMDcoeffs = AMD_stability_coefficients(sim)
    features["C_AMD12"] = AMDcoeffs[0]
    features["C_AMD23"] = AMDcoeffs[1]
    features["C_AMD_max"] = np.max(AMDcoeffs)

    a = np.zeros((sim.N,Nout))
    e = np.zeros((sim.N,Nout))
    inc = np.zeros((sim.N,Nout))
    
    beta12 = np.zeros(Nout)
    beta23 = np.zeros(Nout)

    Rhill12 = ps[1].a*((ps[1].m+ps[2].m)/3.)**(1./3.)
    Rhill23 = ps[2].a*((ps[2].m+ps[3].m)/3.)**(1./3.)
    
    eHill = [0, Rhill12/ps[1].a, max(Rhill12, Rhill23)/ps[2].a, Rhill23/ps[3].a]
    daOvera = [0, (ps[2].a-ps[1].a)/ps[1].a, min(ps[3].a-ps[2].a, ps[2].a-ps[1].a)/ps[2].a, (ps[3].a-ps[2].a)/ps[3].a]
    
    for i, t in enumerate(times):
        for j in [1,2,3]:
            a[j,i] = ps[j].a
            e[j,i] = ps[j].e
            inc[j,i] = ps[j].inc

        # mutual hill radii since that's what goes into Hill stability
        Rhill12 = ps[1].a*((ps[1].m+ps[2].m)/3.)**(1./3.)
        Rhill23 = ps[2].a*((ps[2].m+ps[3].m)/3.)**(1./3.)
        
        beta12[i] = (ps[2].a - ps[1].a)/Rhill12
        beta23[i] = (ps[3].a - ps[2].a)/Rhill23   
        try:
            sim.integrate(t, exact_finish_time=0)
        except:
            break
    
    features['t_final_short'] = sim.t/P0
    
    for string, feature in [("beta12", beta12), ("beta23", beta23)]:
        mean = feature.mean()
        std = feature.std()
        features["avg_"+string] = mean
        features["std_"+string] = std
        features["min_"+string] = min(feature)
        features["max_"+string] = max(feature)

    
    for j in [1,2,3]:
        for string, feature in [('a', a), ('e', e), ('inc', inc)]:
            mean = feature[j].mean()
            std = feature[j].std()
            features['avg_'+string+str(j)] = mean
            features['std_'+string+str(j)] = std
            features['max_'+string+str(j)] = feature[j].max()
            features['min_'+string+str(j)] = feature[j].min()
            features['norm_std_'+string+str(j)] = std/mean
            features['norm_max_'+string+str(j)] = np.abs(feature[j] - mean).max()/mean
            sample = feature[j][:window]
            samplemean = sample.mean()
            features['norm_std_window'+str(window)+'_'+string+str(j)] = sample.std()/samplemean
            features['norm_max_window'+str(window)+'_'+string+str(j)] = np.abs(sample - samplemean).max()/samplemean

        for string, feature in [('eH', e), ('iH', inc)]:
            mean = feature[j].mean()
            std = feature[j].std()

            features['avg_'+string+str(j)] = mean/eHill[j]
            features['std_'+string+str(j)] = std/eHill[j]
            features['max_'+string+str(j)] = feature[j].max()/eHill[j]
            features['min_'+string+str(j)] = feature[j].min()/eHill[j]

        string, feature = ('ecross', e)
        features['avg_'+string+str(j)] = mean/daOvera[j]
        features['std_'+string+str(j)] = std/daOvera[j]
        features['max_'+string+str(j)] = feature[j].max()/daOvera[j]
        features['min_'+string+str(j)] = feature[j].min()/daOvera[j]

        xx = range(a[j].shape[0])
        yy = a[j]/a[j].mean()/features["t_final_short"]
        par = np.polyfit(xx, yy, 1, full=True)
        features['norm_a'+str(j)+'_slope'] = par[0][0]


    return pd.Series(features, index=list(features.keys()))

def findres(sim, i1, i2):
    delta = 0.03
    maxorder = 2
    ps = Poincare.from_Simulation(sim=sim).particles # get averaged mean motions
    n1 = ps[i1].n
    n2 = ps[i2].n
    
    m1 = ps[i1].m
    m2 = ps[i2].m
    
    Pratio = n2/n1
    if np.isnan(Pratio): # probably due to close encounter where averaging step doesn't converge 
        return np.nan, np.nan, np.nan

    res = resonant_period_ratios(Pratio-delta,Pratio+delta, order=maxorder)
    
    Z = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    
    maxstrength = 0
    j, k, i1, i2, strength = -1, -1, -1, -1, -1
    for a, b in res:
        s = np.abs(np.sqrt(m1+m2)*Z**((b-a)/2.)/(b*n2 - a*n1))
        #print('{0}:{1}'.format(b, a), (b*n2 - a*n1), s)
        if s > maxstrength:
            j = b
            k = b-a
            i1 = 1
            i2 = 2
            strength=s
            maxstrength = s
            
    return j, k, strength

def findres2(sim, i1, i2):
    maxorder = 2
    ps = Poincare.from_Simulation(sim=sim).particles # get averaged mean motions
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[i1].M
    m2 = ps[i2].m/ps[i2].M

    Pratio = n2/n1
    if np.isnan(Pratio): # probably due to close encounter where averaging step doesn't converge 
        return np.nan, np.nan, np.nan

    delta = 0.03
    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.999) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=2)
    
    Z = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    Zcross = (ps[i2].a-ps[i1].a)/ps[i1].a
        
    j, k, i1, i2, maxstrength = -1, -1, -1, -1, -1
    for a, b in res:
        s = np.abs(np.sqrt(m1+m2)*(Z/Zcross)**((b-a)/2.)/((b*n2 - a*n1)/n1))
        #print('{0}:{1}'.format(b, a), (b*n2 - a*n1), s)
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s
    
    if maxstrength > -1:
        return j, k, maxstrength
    else:
        return np.nan, np.nan, np.nan

def findresv3(sim, i1, i2):
    maxorder = 2
    try:
        ps = Poincare.from_Simulation(sim=sim, average=False).particles # get averaged mean motions
    except:
        return np.nan, np.nan, np.nan

    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[i1].M
    m2 = ps[i2].m/ps[i2].M

    Pratio = n2/n1
    if np.isnan(Pratio): # probably due to close encounter where averaging step doesn't converge
        return np.nan, np.nan, np.nan

    delta = 0.03
    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.999) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=2)

    Z = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    Zcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, i1, i2, maxstrength = -1, -1, -1, -1, -1
    for a, b in res:
        s = np.abs(np.sqrt(m1+m2)*(Z/Zcross)**((b-a)/2.)/((b*n2 - a*n1)/n1))
        #print('{0}:{1}'.format(b, a), (b*n2 - a*n1), s)
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s

    return j, k, maxstrength

@safe_run_func
def normressummaryfeaturesxgb(sim, args):
    ps = sim.particles
    Mstar = ps[0].m
    P1 = ps[1].P

    sim2 = rebound.Simulation()
    sim2.G = 4*np.pi**2
    sim2.add(m=1.)

    for p in ps[1:]: 
        sim2.add(m=p.m/Mstar, P=p.P/P1, e=p.e, inc=p.inc, pomega=p.pomega, Omega=p.Omega, theta=p.theta)

    sim2.move_to_com()
    sim2.integrator="whfast"
    sim2.dt=sim2.particles[1].P*2.*np.sqrt(3)/100.
    return ressummaryfeaturesxgb(sim2, args)

@safe_run_func
def ressummaryfeaturesxgb(sim, args):
    Norbits = args[0]
    Nout = args[1]
    
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################

    features = OrderedDict()
    try:
        AMDcoeffs = AMD_stability_coefficients(sim)
        features["C_AMD12"] = AMDcoeffs[0]
        features["C_AMD23"] = AMDcoeffs[1]
        features["C_AMD_max"] = np.max(AMDcoeffs)
    except:
        features["C_AMD12"] = np.nan
        features["C_AMD23"] = np.nan
        features["C_AMD_max"] = np.nan

    ps = sim.particles
    sim.init_megno(seed=0)
    
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)
    js, ks, strengths = np.zeros(Npairs), np.zeros(Npairs), np.zeros(Npairs)
    maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = -1, -1, -1, -1, -1, -1

    Zcross = np.zeros(Npairs)
    #print('pairindex, i1, i2, j, k, strength')
    for i, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
        js[i], ks[i], strengths[i] = findresv3(sim, i1, i2)
        Zcross[i] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        #print(i, i1, i2, js[i], ks[i], strengths[i])
        if strengths[i] > maxstrength:
            maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = js[i], ks[i], i1, i2, i, strengths[i]
    
    features['Zcross12'] = Zcross[0]
    features['Zcross13'] = Zcross[1]
    features['Zcross23'] = Zcross[2]
    features['maxj'] = maxj
    features['maxk'] = maxk
    features['maxi1'] = maxi1
    features['maxi2'] = maxi2
    features['maxstrength'] = maxstrength
    
    sortedstrengths = strengths.copy()
    sortedstrengths.sort() # ascending
    if sortedstrengths[-1] > 0 and sortedstrengths[-2] > 0: # if two strongeest resonances are nonzereo
        features['secondres'] = sortedstrengths[-2]/sortedstrengths[-1] # ratio of strengths
    else:
        features['secondres'] = -1
        
    #print('max', maxi1, maxi2, maxj, maxk, maxpairindex, maxstrength)
    #print('df (j, k, pairindex):', features['j'], features['k'], features['pairindex'])

    times = np.linspace(0, Norbits*sim.particles[1].P, Nout)

    eminus = np.zeros((Npairs, Nout))
    rebound_Z, rebound_phi = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    rebound_Zcom, rebound_phiZcom = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    rebound_Zstar, rebound_dKprime = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    celmech_Z, celmech_phi = np.zeros(Nout), np.zeros(Nout)
    celmech_Zcom, celmech_phiZcom = np.zeros(Nout), np.zeros(Nout)
    celmech_Zstar, celmech_dKprime = np.zeros(Nout), np.zeros(Nout)

    for i,t in enumerate(times):
        for j, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
            i1, i2 = int(i1), int(i2)
            eminus[j, i] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2)
            if js[j] != -1:
                pvars = Poincare.from_Simulation(sim, average=False)
                avars = Andoyer.from_Poincare(pvars, j=int(js[j]), k=int(ks[j]), a10=a0[i1], i1=i1, i2=i2)
                rebound_Z[j, i] = avars.Z
                rebound_phi[j, i] = avars.phi
                rebound_Zcom[j, i] = avars.Zcom
                rebound_phiZcom[j, i] = avars.phiZcom
                rebound_Zstar[j, i] = avars.Zstar
                rebound_dKprime[j, i] = avars.dKprime
        try:
            sim.integrate(t, exact_finish_time=0)
        except:
            break
        
    mask = eminus[0] > 0 # where there are data points in case sim ends early
    times = times[mask]
    eminus = eminus[:, mask]
    rebound_Z, rebound_phi = rebound_Z[:, mask], rebound_phi[:, mask]
    rebound_Zcom, rebound_phiZcom =  rebound_Zcom[:, mask], rebound_phiZcom[:, mask]
    rebound_Zstar, rebound_dKprime = rebound_Zstar[:, mask], rebound_dKprime[:, mask]
    celmech_Z, celmech_phi, celmech_Zcom, celmech_phiZcom = celmech_Z[mask], celmech_phi[mask], celmech_Zcom[mask], celmech_phiZcom[mask]
    celmech_Zstar, celmech_dKprime = celmech_Zstar[mask], celmech_dKprime[mask]
    
    for i, s in zip([0,2], ['12', '23']): # take adjacent ones
        EM = eminus[i]
        Zc = Zcross[i]
        features['EMmed'+s] = np.median(EM)/Zc
        features['EMmax'+s] = EM.max()/Zc
        try:
            p = np.poly1d(np.polyfit(times, EM, 3))
            m = p(times)
            EMdrift = np.abs((m[-1]-m[0])/m[0])
            features['EMdrift'+s] = EMdrift
        except:
            features['EMdrift'+s] = np.nan
        maxindex = (m == m.max()).nonzero()[0][0] # index where cubic polynomial fit to EM reaches max to track long wavelength variations (secular?)
        if EMdrift > 0.1 and (maxindex < 0.01*Nout or maxindex > 0.99*Nout): # don't flag as not capturing secular if Z isn't varying significantly in first place
            features['capseculartscale'+s] = 0
        else:
            features['capseculartscale'+s] = 1
        features['EMdetrendedstd'+s] = pd.Series(EM-m).std()/EM[0]
        rollstd = pd.Series(EM).rolling(window=100).std()
        features['EMrollingstd'+s] = rollstd[100:].median()/EM[0]
        var = [EM[:j].var() for j in range(len(EM))]
        try:
            p = np.poly1d(np.polyfit(times[len(var)//2:], var[len(var)//2:], 1)) # fit only second half to get rid of transient
            features['DiffcoeffEM'+s] = p[1]/Zc**2
        except:
            features['DiffcoeffEM'+s] = np.nan
        features['medvarEM'+s] = np.median(var[len(var)//2:])/Zc**2
        if strengths[i] != -1:
            Z = rebound_Z[i]
            features['Zmed'+s] = np.median(Z)/Zc
            features['Zmax'+s] = rebound_Z[i].max()/Zc
            try:
                p = np.poly1d(np.polyfit(times, Z, 3))
                m = p(times)
                features['Zdetrendedstd'+s] = pd.Series(Z-m).std()/Z[0]
            except:
                features['Zdetrendedstd'+s] = np.nan
            rollstd = pd.Series(Z).rolling(window=100).std()
            features['Zrollingstd'+s] = rollstd[100:].median()/Z[0]
            var = [Z[:j].var() for j in range(len(Z))]
            try:
                p = np.poly1d(np.polyfit(times[len(var)//2:], var[len(var)//2:], 1)) # fit only second half to get rid of transient
                features['DiffcoeffZ'+s] = p[1]/Zc**2
            except:
                features['DiffcoeffZ'+s] = np.nan
            features['medvarZ'+s] = np.median(var[len(var)//2:])/Zc**2
            features['Zcomdrift'+s] = np.max(np.abs(rebound_Zcom[i]-rebound_Zcom[i, 0])/rebound_Zcom[i, 0])
            rollstd = pd.Series(rebound_Zcom[i]).rolling(window=100).std()
            features['Zcomrollingstd'+s] = rollstd[100:].median()/rebound_Zcom[i,0]
            features['phiZcomdrift'+s] = np.max(np.abs(rebound_phiZcom[i]-rebound_phiZcom[i, 0]))
            rollstd = pd.Series(rebound_phiZcom[i]).rolling(window=100).std()
            features['phiZcomrollingstd'+s] = rollstd[100:].median()
            features['Zstardrift'+s] = np.max(np.abs(rebound_Zstar[i]-rebound_Zstar[i, 0])/rebound_Zstar[i, 0])
            rollstd = pd.Series(rebound_Zstar[i]).rolling(window=100).std()
            features['Zstarrollingstd'+s] = rollstd[100:].median()/rebound_Zstar[i,0]
            Zcosphi = Z*np.cos(rebound_phi[i])
            features['Zcosphistd'+s] = Zcosphi.std()/Zc
            features['medZcosphi'+s] = np.median(Zcosphi)/Zc
        else:
            features['Zmed'+s] = -1
            features['Zmax'+s] = -1
            features['Zdetrendedstd'+s] = -1
            features['Zrollingstd'+s] = -1
            features['DiffcoeffZ'+s] = -1
            features['medvarZ'+s] = -1
            features['Zcomdrift'+s] = -1
            features['Zcomrollingstd'+s] = -1
            features['phiZcomdrift'+s] = -1
            features['phiZcomrollingstd'+s] = -1
            features['Zstardrift'+s] = -1
            features['Zstarrollingstd'+s] = -1
            features['Zcosphistd'+s] = -1
            features['medZcosphi'+s] = -1

    tlyap = 1./np.abs(sim.calculate_lyapunov())
    if tlyap > Norbits:
        tlyap = Norbits
    features['tlyap'] = tlyap
    features['megno'] = sim.calculate_megno()
    return pd.Series(features, index=list(features.keys()))

@safe_run_func
def ressummaryfeaturesxgb2(sim, args):
    Norbits = args[0]
    Nout = args[1]
    
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################

    features = OrderedDict()
    AMDcoeffs = AMD_stability_coefficients(sim)
    features["C_AMD12"] = AMDcoeffs[0]
    features["C_AMD23"] = AMDcoeffs[1]
    features["C_AMD_max"] = np.max(AMDcoeffs)

    ps = sim.particles
    sim.init_megno()
    
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)
    js, ks, strengths = np.zeros(Npairs, dtype=np.int), np.zeros(Npairs, dtype=np.int), np.zeros(Npairs)
    maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = -1, -1, -1, -1, -1, -1

    Zcross = np.zeros(Npairs)
    #print('pairindex, i1, i2, j, k, strength')
    for i, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
        js[i], ks[i], strengths[i] = findresv3(sim, i1, i2)
        Zcross[i] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        #print(i, i1, i2, js[i], ks[i], strengths[i])
        if strengths[i] > maxstrength:
            maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = js[i], ks[i], i1, i2, i, strengths[i]
    
    features['Zcross12'] = Zcross[0]
    features['Zcross13'] = Zcross[1]
    features['Zcross23'] = Zcross[2]
    features['maxj'] = maxj
    features['maxk'] = maxk
    features['maxi1'] = maxi1
    features['maxi2'] = maxi2
    features['maxstrength'] = maxstrength
    
    sortedstrengths = strengths.copy()
    sortedstrengths.sort()
    if sortedstrengths[-1] > 0 and sortedstrengths[-2] > 0:
        features['secondres'] = sortedstrengths[-2]/sortedstrengths[-1]
    else:
        features['secondres'] = -1
        
    #print('max', maxi1, maxi2, maxj, maxk, maxpairindex, maxstrength)
    #print('df (j, k, pairindex):', features['j'], features['k'], features['pairindex'])
    P0 = sim.particles[1].P
    times = np.linspace(0, Norbits, Nout)

    eminus = np.zeros((Npairs, Nout))
    rebound_Z, rebound_phi = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    rebound_Zcom, rebound_phiZcom = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    rebound_Zstar, rebound_dKprime = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    celmech_Z, celmech_phi = np.zeros(Nout), np.zeros(Nout)
    celmech_Zcom, celmech_phiZcom = np.zeros(Nout), np.zeros(Nout)
    celmech_Zstar, celmech_dKprime = np.zeros(Nout), np.zeros(Nout)

    for i,t in enumerate(times):
        for j, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
            i1, i2 = int(i1), int(i2)
            eminus[j, i] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2)
            if js[j] != -1:
                pvars = Poincare.from_Simulation(sim)
                avars = Andoyer.from_Poincare(pvars, j=js[j], k=ks[j], a10=a0[i1], i1=i1, i2=i2)
                rebound_Z[j, i] = avars.Z
                rebound_phi[j, i] = avars.phi
                rebound_Zcom[j, i] = avars.Zcom
                rebound_phiZcom[j, i] = avars.phiZcom
                rebound_Zstar[j, i] = avars.Zstar
                rebound_dKprime[j, i] = avars.dKprime
        try:
            sim.integrate(t*P0, exact_finish_time=0)
        except:
            break
        
    mask = eminus[0] > 0 # where there are data points in case sim ends early
    times = times[mask]
    eminus = eminus[:, mask]
    rebound_Z, rebound_phi = rebound_Z[:, mask], rebound_phi[:, mask]
    rebound_Zcom, rebound_phiZcom =  rebound_Zcom[:, mask], rebound_phiZcom[:, mask]
    rebound_Zstar, rebound_dKprime = rebound_Zstar[:, mask], rebound_dKprime[:, mask]
    celmech_Z, celmech_phi, celmech_Zcom, celmech_phiZcom = celmech_Z[mask], celmech_phi[mask], celmech_Zcom[mask], celmech_phiZcom[mask]
    celmech_Zstar, celmech_dKprime = celmech_Zstar[mask], celmech_dKprime[mask]
    
    for i, s in zip([0,2], ['12', '23']): # take adjacent ones
        EM = eminus[i]
        Zc = Zcross[i]
        features['EMmed'+s] = np.median(EM)/Zc
        features['EMmax'+s] = EM.max()/Zc
        try:
            p = np.poly1d(np.polyfit(times, EM, 3))
            m = p(times)
            EMdrift = np.abs((m[-1]-m[0])/m[0])
            features['EMdrift'+s] = EMdrift
        except:
            features['EMdrift'+s] = np.nan
        maxindex = (m == m.max()).nonzero()[0][0] # index where cubic polynomial fit to EM reaches max to track long wavelength variations (secular?)
        if EMdrift > 0.1 and (maxindex < 0.01*Nout or maxindex > 0.99*Nout): # don't flag as not capturing secular if Z isn't varying significantly in first place
            features['capseculartscale'+s] = 0
        else:
            features['capseculartscale'+s] = 1
        features['EMdetrendedstd'+s] = pd.Series(EM-m).std()/EM[0]
        rollstd = pd.Series(EM).rolling(window=100).std()
        features['EMrollingstd'+s] = rollstd[100:].median()/EM[0]
        var = [EM[:j].var() for j in range(len(EM))]
        try:
            p = np.poly1d(np.polyfit(times[len(var)//2:], var[len(var)//2:], 1)) # fit only second half to get rid of transient
            features['DiffcoeffEM'+s] = p[1]/Zc**2
        except:
            features['DiffcoeffEM'+s] = np.nan
        features['medvarEM'+s] = np.median(var[len(var)//2:])/Zc**2
        if strengths[i] != -1:
            Z = rebound_Z[i]
            features['Zmed'+s] = np.median(Z)/Zc
            features['Zmax'+s] = rebound_Z[i].max()/Zc
            try:
                p = np.poly1d(np.polyfit(times, Z, 3))
                m = p(times)
                features['Zdetrendedstd'+s] = pd.Series(Z-m).std()/Z[0]
            except:
                features['Zdetrendedstd'+s] = np.nan
            rollstd = pd.Series(Z).rolling(window=100).std()
            features['Zrollingstd'+s] = rollstd[100:].median()/Z[0]
            var = [Z[:j].var() for j in range(len(Z))]
            try:
                p = np.poly1d(np.polyfit(times[len(var)//2:], var[len(var)//2:], 1)) # fit only second half to get rid of transient
                features['DiffcoeffZ'+s] = p[1]/Zc**2
            except:
                features['DiffcoeffZ'+s] = np.nan
            features['medvarZ'+s] = np.median(var[len(var)//2:])/Zc**2
            features['Zcomdrift'+s] = np.max(np.abs(rebound_Zcom[i]-rebound_Zcom[i, 0])/rebound_Zcom[i, 0])
            rollstd = pd.Series(rebound_Zcom[i]).rolling(window=100).std()
            features['Zcomrollingstd'+s] = rollstd[100:].median()/rebound_Zcom[i,0]
            features['phiZcomdrift'+s] = np.max(np.abs(rebound_phiZcom[i]-rebound_phiZcom[i, 0]))
            rollstd = pd.Series(rebound_phiZcom[i]).rolling(window=100).std()
            features['phiZcomrollingstd'+s] = rollstd[100:].median()
            features['Zstardrift'+s] = np.max(np.abs(rebound_Zstar[i]-rebound_Zstar[i, 0])/rebound_Zstar[i, 0])
            rollstd = pd.Series(rebound_Zstar[i]).rolling(window=100).std()
            features['Zstarrollingstd'+s] = rollstd[100:].median()/rebound_Zstar[i,0]
            Zcosphi = Z*np.cos(rebound_phi[i])
            features['Zcosphistd'+s] = Zcosphi.std()/Zc
            features['medZcosphi'+s] = np.median(Zcosphi)/Zc
        else:
            features['Zmed'+s] = -1
            features['Zmax'+s] = -1
            features['Zdetrendedstd'+s] = -1
            features['Zrollingstd'+s] = -1
            features['DiffcoeffZ'+s] = -1
            features['medvarZ'+s] = -1
            features['Zcomdrift'+s] = -1
            features['Zcomrollingstd'+s] = -1
            features['phiZcomdrift'+s] = -1
            features['phiZcomrollingstd'+s] = -1
            features['Zstardrift'+s] = -1
            features['Zstarrollingstd'+s] = -1
            features['Zcosphistd'+s] = -1
            features['medZcosphi'+s] = -1

    tlyap = 1./np.abs(sim.calculate_lyapunov())/P0
    if tlyap > Norbits:
        tlyap = Norbits
    features['tlyap'] = tlyap
    features['megno'] = sim.calculate_megno()
    return pd.Series(features, index=list(features.keys()))

def fillnan(features, pairs):
    features['tlyap'] = np.nan
    features['megno'] = np.nan

    for i, [label, i1, i2] in enumerate(pairs):
        features['EMmed'+label] = np.nan
        features['EMmax'+label] = np.nan
        features['EMstd'+label] = np.nan
        features['EMslope'+label] = np.nan
        features['cheapEMslope'+label] = np.nan
        features['EMrollingstd'+label] = np.nan
        features['EPmed'+label] = np.nan
        features['EPmax'+label] = np.nan
        features['EPstd'+label] = np.nan
        features['EPslope'+label] = np.nan
        features['cheapEPslope'+label] = np.nan
        features['EProllingstd'+label] = np.nan
        features['Zstarslope'+label] = np.nan
        features['Zstarrollingstd'+label] = np.nan
        features['Zcommed'+label] = np.nan
        features['Zfree'+label] = np.nan
        features['Zstarmed'+label] = np.nan
        features['Zstarstd'+label] = np.nan
        features['Zstarmed'+label] = np.nan
        features['Zstarslope'+label] = np.nan
        features['cheapZstarslope'+label] = np.nan

def fillnanv5(features, pairs):
    features['tlyap'] = np.nan
    features['megno'] = np.nan

    for i, [label, i1, i2] in enumerate(pairs):
        features['EMmed'+label] = np.nan
        features['EMmax'+label] = np.nan
        features['EMstd'+label] = np.nan
        features['EMslope'+label] = np.nan
        features['EMrollingstd'+label] = np.nan
        features['EPmed'+label] = np.nan
        features['EPmax'+label] = np.nan
        features['EPstd'+label] = np.nan
        features['EPslope'+label] = np.nan
        features['EProllingstd'+label] = np.nan
        features['Zstarslope'+label] = np.nan
        features['Zstarrollingstd'+label] = np.nan
        features['Zcommed'+label] = np.nan
        features['Zfree'+label] = np.nan
        features['Zstarmed'+label] = np.nan
        features['Zstarstd'+label] = np.nan
        features['Zstarmed'+label] = np.nan
        features['Zstarslope'+label] = np.nan

def ressummaryfeaturesxgbv4(sim, args):
    Norbits = args[0]
    Nout = args[1]
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################
    ps = sim.particles
    sim.init_megno()
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    features = resparams(sim, args)
    pairs, nearpair = getpairs(sim)
    
    P0 = ps[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    Z, phi = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zcom, phiZcom = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zstar = np.zeros((Npairs,Nout))
    eminus, eplus = np.zeros((Npairs,Nout)), np.zeros((Npairs, Nout))
    
    features['unstableinNorbits'] = False
    AMD0 = 0
    for p in ps[1:sim.N-sim.N_var]:
        AMD0 += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))

    AMDerr = np.zeros(Nout)
    for i,t in enumerate(times):
        try:
            sim.integrate(t*P0, exact_finish_time=0)
        except:
            features['unstableinNorbits'] = True
            break
        AMD = 0
        for p in ps[1:sim.N-sim.N_var]:
            AMD += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))
        AMDerr[i] = np.abs((AMD-AMD0)/AMD0)
        
        for j, [label, i1, i2] in enumerate(pairs):
            #i1 = int(i1); i2 = int(i2)
            eminus[j, i] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2) / features['EMcross'+label]
            eplus[j, i] = np.sqrt((ps[i1].m*ps[i1].e*np.cos(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].m*ps[i1].e*np.sin(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.sin(ps[i2].pomega))**2)/(ps[i1].m+ps[i2].m)
            if features['strength'+label] > 0:
                pvars = Poincare.from_Simulation(sim)
                avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
            
                Z[j, i] = avars.Z / features['Zcross'+label]
                phi[j, i] = avars.phi
                Zcom[j, i] = avars.Zcom
                phiZcom[j, i] = avars.phiZcom
                Zstar[j, i] = avars.Zstar / features['Zcross'+label]
            
    fillnan(features, pairs)
    
    Nnonzero = int((eminus[0,:] > 0).sum())
    times = times[:Nnonzero]
    AMDerr = AMDerr[:Nnonzero]
    
    eminus = eminus[:,:Nnonzero]
    eplus = eplus[:,:Nnonzero]
    Z = Z[:,:Nnonzero]
    phi = phi[:,:Nnonzero]
    Zcom = Zcom[:,:Nnonzero]
    phiZcom = phiZcom[:,:Nnonzero]
    Zstar = Zstar[:,:Nnonzero]
    
    # Features with or without resonances:
    tlyap = 1./np.abs(sim.calculate_lyapunov())/P0
    if tlyap > Norbits:
        tlyap = Norbits
    features['tlyap'] = tlyap
    features['megno'] = sim.calculate_megno()
    features['AMDerr'] = AMDerr.max()
    
    for i, [label, i1, i2] in enumerate(pairs):
        EM = eminus[i,:]
        EP = eplus[i,:]
        features['EMmed'+label] = np.median(EM)
        features['EMmax'+label] = EM.max()
        features['EMstd'+label] = EM.std()
        features['EPmed'+label] = np.median(EP)
        features['EPmax'+label] = EP.max()
        features['EPstd'+label] = EP.std()
        
        try:
            m,c  = np.linalg.lstsq(np.vstack([times/P0, np.ones(len(times))]).T, EM)[0]
            features['EMslope'+label] = m # EM/EMcross per orbit so we can compare different length integrations
            last = np.median(EM[-int(Nout/20):])
            features['cheapEMslope'+label] = (last - EM.min())/EM.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
        except:
            pass
        
        rollstd = pd.Series(EM).rolling(window=10).std()
        features['EMrollingstd'+label] = rollstd[10:].median()/features['EMmed'+label]
        
        try:
            m,c  = np.linalg.lstsq(np.vstack([times/P0, np.ones(len(times))]).T, EP)[0]
            features['EPslope'+label] = m
            last = np.median(EP[-int(Nout/20):])
            features['cheapEPslope'+label] = (last - EP.min())/EP.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
        except:
            pass
            
        rollstd = pd.Series(EP).rolling(window=10).std()
        features['EProllingstd'+label] = rollstd[10:].median()/features['EPmed'+label]
        
        if features['strength'+label] > 0:
            features['Zcommed'+label] = np.median(Zcom[i,:])
            
                
            if np.median(Zstar[i,:]) > 0:
                ZS = Zstar[i,:]
                features['Zstarmed'+label] = np.median(ZS)
                # try this one generically for variation in constants, since all fixed points seem to follow Zstar closely?
                features['Zstarstd'+label] = Zstar[i,:].std()/features['Zstarmed'+label]
                try:
                    m,c  = np.linalg.lstsq(np.vstack([times/P0, np.ones(len(times))]).T, ZS)[0]
                    features['Zstarslope'+label] = m
                    last = np.median(ZS[-int(Nout/20):])
                    features['cheapZstarslope'+label] = (last - ZS.min())/ZS.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
                except:
                    pass
    
                rollstd = pd.Series(ZS).rolling(window=10).std()
                features['Zstarrollingstd'+label] = rollstd[10:].median()/features['Zstarmed'+label]
               
                Zx = Z[i,:]*np.cos(phi[i,:])
                Zy = Z[i,:]*np.sin(phi[i,:]) 
                Zfree = np.sqrt((Zx+Zstar)**2 + Zy**2) # Zstar at (-Zstar, 0)
                features['Zfree'+label] = Zfree.std()/features['reshalfwidth'+label] # free Z around Zstar
                
    return pd.Series(features, index=list(features.keys())) 

def ressummaryfeaturesxgbv5(sim, args):
    Norbits = args[0]
    Nout = args[1]
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################
    ps = sim.particles
    sim.init_megno()
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    features = resparams(sim, args)
    pairs = getpairsv5(sim)
    
    P0 = ps[1].P
    times = np.linspace(0, Norbits*P0, Nout)
    Z, phi = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zcom, phiZcom = np.zeros((Npairs,Nout)), np.zeros((Npairs,Nout))
    Zstar = np.zeros((Npairs,Nout))
    eminus, eplus = np.zeros((Npairs,Nout)), np.zeros((Npairs, Nout))
    
    features['unstableinNorbits'] = False
    AMD0 = 0
    for p in ps[1:sim.N-sim.N_var]:
        AMD0 += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))

    AMDerr = np.zeros(Nout)
    for i,t in enumerate(times):
        try:
            sim.integrate(t*P0, exact_finish_time=0)
        except:
            features['unstableinNorbits'] = True
            break
        AMD = 0
        for p in ps[1:sim.N-sim.N_var]:
            AMD += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))
        AMDerr[i] = np.abs((AMD-AMD0)/AMD0)
        
        for j, [label, i1, i2] in enumerate(pairs):
            Zcross = features['EMcross'+label]/np.sqrt(2) # important factor of sqrt(2)!
            #i1 = int(i1); i2 = int(i2)
            eminus[j, i] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2) / features['EMcross'+label]
            eplus[j, i] = np.sqrt((ps[i1].m*ps[i1].e*np.cos(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].m*ps[i1].e*np.sin(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.sin(ps[i2].pomega))**2)/(ps[i1].m+ps[i2].m)
            if features['strength'+label] > 0:
                pvars = Poincare.from_Simulation(sim)
                avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
            
                Z[j, i] = avars.Z / Zcross
                phi[j, i] = avars.phi
                Zcom[j, i] = avars.Zcom
                phiZcom[j, i] = avars.phiZcom
                Zstar[j, i] = avars.Zstar / Zcross
            
    fillnanv5(features, pairs)
    
    Nnonzero = int((eminus[0,:] > 0).sum())
    times = times[:Nnonzero]
    AMDerr = AMDerr[:Nnonzero]
    
    eminus = eminus[:,:Nnonzero]
    eplus = eplus[:,:Nnonzero]
    Z = Z[:,:Nnonzero]
    phi = phi[:,:Nnonzero]
    Zcom = Zcom[:,:Nnonzero]
    phiZcom = phiZcom[:,:Nnonzero]
    Zstar = Zstar[:,:Nnonzero]
    
    # Features with or without resonances:
    tlyap = 1./np.abs(sim.calculate_lyapunov())/P0
    if tlyap > Norbits:
        tlyap = Norbits
    features['tlyap'] = tlyap
    features['megno'] = sim.calculate_megno()
    features['AMDerr'] = AMDerr.max()
    
    for i, [label, i1, i2] in enumerate(pairs):
        EM = eminus[i,:]
        EP = eplus[i,:]
        features['EMmed'+label] = np.median(EM)
        features['EMmax'+label] = EM.max()
        features['EMstd'+label] = EM.std()
        features['EPmed'+label] = np.median(EP)
        features['EPmax'+label] = EP.max()
        features['EPstd'+label] = EP.std()
        
        last = np.median(EM[-int(Nout/20):])
        features['EMslope'+label] = (last - EM.min())/EM.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
        
        rollstd = pd.Series(EM).rolling(window=10).std()
        features['EMrollingstd'+label] = rollstd[10:].median()/features['EMmed'+label]
        
        last = np.median(EP[-int(Nout/20):])
        features['EPslope'+label] = (last - EP.min())/EP.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
            
        rollstd = pd.Series(EP).rolling(window=10).std()
        features['EProllingstd'+label] = rollstd[10:].median()/features['EPmed'+label]
        
        if features['strength'+label] > 0:
            features['Zcommed'+label] = np.median(Zcom[i,:])
            
                
            if np.median(Zstar[i,:]) > 0:
                ZS = Zstar[i,:]
                features['Zstarmed'+label] = np.median(ZS)
                # try this one generically for variation in constants, since all fixed points seem to follow Zstar closely?
                features['Zstarstd'+label] = Zstar[i,:].std()/features['Zstarmed'+label]
                last = np.median(ZS[-int(Nout/20):])
                features['Zstarslope'+label] = (last - ZS.min())/ZS.std() # measure of whether final value of EM is much higher than minimum compared to std to test whether we captured long timescale
    
                rollstd = pd.Series(ZS).rolling(window=10).std()
                features['Zstarrollingstd'+label] = rollstd[10:].median()/features['Zstarmed'+label]
               
                Zx = Z[i,:]*np.cos(phi[i,:])
                Zy = Z[i,:]*np.sin(phi[i,:]) 
                Zfree = np.sqrt((Zx+Zstar)**2 + Zy**2) # Zstar at (-Zstar, 0)
                features['Zfree'+label] = Zfree.std()#/features['reshalfwidth'+label] # free Z around Zstar
                #For k=1 can always define a Zstar, even w/o separatrix, but here reshalfwidth is nan. So don't  normalize
                
    return pd.Series(features, index=list(features.keys())) 

def getpairs(sim):
    N = sim.N - sim.N_var
    Npairs = int((N-1)*(N-2)/2)
    EMcross = np.zeros(Npairs)
    ps = sim.particles
    #print('pairindex, i1, i2, j, k, strength')
    for i, [i1, i2] in enumerate(itertools.combinations(np.arange(1, N), 2)):
        i1 = int(i1); i2 = int(i2)
        EMcross[i] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        
    if EMcross[0] < EMcross[2]: # 0 = '1-2', 2='2-3'
        nearpair = 12 # for visualization and debugging
        pairs = [['near', 1,2], ['far', 2, 3], ['outer', 1, 3]]
    else:
        nearpair = 23
        pairs = [['near', 2, 3], ['far', 1, 2], ['outer', 1, 3]]

    return pairs, nearpair 

def getpairsv5(sim, indices=[1, 2, 3]):
    ps = sim.particles
    sortedindices = sorted(indices, key=lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = (ps[sortedindices[1]].a-ps[sortedindices[0]].a)/ps[sortedindices[0]].a
    EMcrossOuter = (ps[sortedindices[2]].a-ps[sortedindices[1]].a)/ps[sortedindices[1]].a

    if EMcrossInner < EMcrossOuter:
        return [
            ['near', sortedindices[0], sortedindices[1]],
            ['far', sortedindices[1], sortedindices[2]] ,
            ['outer', sortedindices[0], sortedindices[2]]
        ]
    else:
        return [
            ['near', sortedindices[1], sortedindices[2]],
            ['far', sortedindices[0], sortedindices[1]],
            ['outer', sortedindices[0], sortedindices[2]],
        ]

def resparams(sim, args):
    Norbits = args[0]
    Nout = args[1]

    features = OrderedDict()
    ps = sim.particles
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    pairs, features['nearpair'] = getpairs(sim)
    maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = -1, -1, -1, -1, -1, -1
    for i, [label, i1, i2] in enumerate(pairs):
        # recalculate with new ordering
        RH = ps[i1].a*((ps[i1].m + ps[i2].m)/ps[0].m)**(1./3.)
        features['beta'+label] = (ps[i2].a-ps[i1].a)/RH
        features['EMcross'+label] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        features['Zcross'+label] = features['EMcross'+label] / np.sqrt(2) # IMPORTANT FACTOR OF SQRT(2)! Z = EM/sqrt(2)
        
        features['j'+label], features['k'+label], features['strength'+label] = findresv3(sim, i1, i2) # returns -1s if no res found
        if features['strength'+label] > maxstrength:
            maxj, maxk, maxi1, maxi2, maxpairindex, maxstrength = features['j'+label], features['k'+label], i1, i2, i, features['strength'+label]
        features["C_AMD"+label] = AMD_stability_coefficient(sim, i1, i2)
        
        features['reshalfwidth'+label] = np.nan # so we always populate in case there's no  separatrix
        if features['strength'+label] > 0:
            pvars = Poincare.from_Simulation(sim)
            avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
            Zsepinner = avars.Zsep_inner # always positive, location at (-Zsepinner, 0)
            Zsepouter = avars.Zsep_outer
            Zstar = avars.Zstar
            features['reshalfwidth'+label] = min(Zsepouter-Zstar, Zstar-Zsepinner)
    
    features['maxj'] = maxj
    features['maxk'] = maxk
    features['maxi1'] = maxi1
    features['maxi2'] = maxi2
    features['maxstrength'] = maxstrength
    sortedstrengths = np.array([features['strength'+label] for label in ['near', 'far', 'outer']])
    sortedstrengths.sort() # ascending
    
    if sortedstrengths[-1] > 0 and sortedstrengths[-2] > 0: # if two strongeest resonances are nonzereo
        features['secondres'] = sortedstrengths[-2]/sortedstrengths[-1] # ratio of second largest strength to largest
    else:
        features['secondres'] = -1
            
    return pd.Series(features, index=list(features.keys())) 

def resparamsv5(sim, args, trio):
    Norbits = args[0]
    Nout = args[1]

    features = OrderedDict()
    ps = sim.particles
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    Npairs = int((N-1)*(N-2)/2)

    pairs = getpairsv5(sim, trio)
    for i, [label, i1, i2] in enumerate(pairs):
        # recalculate with new ordering
        RH = ps[i1].a*((ps[i1].m + ps[i2].m)/ps[0].m)**(1./3.)
        features['beta'+label] = (ps[i2].a-ps[i1].a)/RH
        features['EMcross'+label] = (ps[int(i2)].a-ps[int(i1)].a)/ps[int(i1)].a
        features['j'+label], features['k'+label], features['strength'+label] = findresv3(sim, i1, i2) # returns -1s if no res found
        features["C_AMD"+label] = AMD_stability_coefficient(sim, i1, i2)
        features['reshalfwidth'+label] = np.nan # so we always populate in case there's no  separatrix
        if features['strength'+label] > 0:
            pvars = Poincare.from_Simulation(sim, average=False)
            avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
            Zsepinner = avars.Zsep_inner # always positive, location at (-Zsepinner, 0)
            Zsepouter = avars.Zsep_outer
            Zstar = avars.Zstar
            features['reshalfwidth'+label] = min(Zsepouter-Zstar, Zstar-Zsepinner)
    
    sortedstrengths = np.array([features['strength'+label] for label in ['near', 'far', 'outer']])
    sortedstrengths.sort() # ascending
    
    if sortedstrengths[-1] > 0 and sortedstrengths[-2] > 0: # if two strongeest resonances are nonzereo
        features['secondres'] = sortedstrengths[-2]/sortedstrengths[-1] # ratio of second largest strength to largest
    else:
        features['secondres'] = -1
            
    return pd.Series(features, index=list(features.keys())) 

def restseriesv5(sim, args, trio): # corresponds to ressummaryfeaturesxgbv5
    # TODO: Need to call get_tseries to see if unstable or not.
    #  See features - call to init_sim.
    Norbits = args[0]
    Nout = args[1]
    ###############################
    sim.collision_resolve = collision
    sim.ri_whfast.keep_unsynchronized = 1
    sim.ri_whfast.safe_mode = 0
    ##############################
    ps = sim.particles
    sim.init_megno()
    
    P0 = ps[1].P
    times = np.linspace(0, Norbits*P0, Nout)
   
    features = resparamsv5(sim, args, trio)
    pairs = getpairsv5(sim, trio)

    AMD0 = 0
    N = sim.N - sim.N_var
    a0 = [0] + [sim.particles[i].a for i in range(1, N)]
    for p in [ps[i] for i in trio]: #TODO: Dan, is this correct? Or should AMD0 be the entire range(1, N)?
        AMD0 += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))

    val = np.zeros((Nout, 27))
    for i, time in enumerate(times):
        try:
            #print(sim.t)
            sim.integrate(time, exact_finish_time=0)
        except:
            break
        AMD = 0
        for p in [ps[i] for i in trio]:
            AMD += p.m*np.sqrt(sim.G*ps[0].m*p.a)*(1-np.sqrt(1-p.e**2)*np.cos(p.inc))
        
        val[i,0] = sim.t/P0  # time

        Ns = 8
        for j, [label, i1, i2] in enumerate(pairs):
            Zcross = features['EMcross'+label]/np.sqrt(2)
            #i1 = int(i1); i2 = int(i2)
            val[i,Ns*j+1] = np.sqrt((ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega))**2 + (ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))**2) / features['EMcross'+label]# eminus
            val[i,Ns*j+2] = np.sqrt((ps[i1].m*ps[i1].e*np.cos(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].m*ps[i1].e*np.sin(ps[i1].pomega) + ps[i2].m*ps[i2].e*np.sin(ps[i2].pomega))**2)/(ps[i1].m+ps[i2].m) # eeplus
            if features['strength'+label] > 0:
                try:
                    pvars = Poincare.from_Simulation(sim, average=False)
                except:
                    return val * np.nan
                avars = Andoyer.from_Poincare(pvars, j=int(features['j'+label]), k=int(features['k'+label]), a10=a0[i1], i1=i1, i2=i2)
           
                Z = avars.Z
                phi = avars.phi
                Zstar = avars.Zstar
                val[i, Ns*j+3] = Z/Zcross
                val[i, Ns*j+4] = phi
                val[i, Ns*j+5] = avars.Zcom
                val[i, Ns*j+6] = avars.phiZcom
                val[i, Ns*j+7] = Zstar/Zcross
                if not np.isnan(Zstar):
                    Zx = Z*np.cos(phi)
                    Zy = Z*np.sin(phi) 
                    Zfree = np.sqrt((Zx+Zstar)**2 + Zy**2) # Zstar at (-Zstar, 0)
                    val[i, Ns*j+8]= Zfree# features['reshalfwidth'+label] # free Z around Zstar
                    #For k=1 can always define a Zstar, even w/o separatrix, but here reshalfwidth is nan. So don't  normalize

        val[i,25] = np.abs((AMD-AMD0)/AMD0) # AMDerr
        val[i,26] = sim.calculate_megno() # megno

    return val
