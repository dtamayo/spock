import numpy as np
import rebound
from .simsetup import init_sim_parameters
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

class NbodyRegressor():
    def __init__(self):
        pass 
    
    def predict_instability_time(self, sim, tmax=None, archive_filename=None, archive_interval=None, n_jobs=-1, match_training=False):
        """
        Predict instability time through N-body integration.

        Parameters:

        sim (rebound.Simulation, or list):   Orbital configuration(s) to test
        tmax (float):               Maximum time to integrate for (in Simulation time units). If passing a list of sims, need to pass a list of tmax of equal length. Defaults to 1e9 innermost planet orbits.
        archive_filename (str):     Path and filename to store a rebound SimulationArchive of snapshots
        archive_interval (float):   Time between snapshots for SimulationArchive (in Simulation time units)
        n_jobs (int):               Number of cores to use for calculation (only if passing more than one simulation). Default: Use all available cores. 
        match_training (bool):      In order to match the exact chaotic trajectory given an input orbital configuration from our training set, need to set match_training=True. False gives an equally valid chaotic realization, and gives a factor of ~2 speedup in evaluation.

        Returns:

        float:  Time of instability (Hill sphere crossing), in Simulation time units, or tmax if stable
        int:    1 if integration reached tmax, 0 if Hill spheres crossed
        """
 
        if isinstance(sim, rebound.Simulation):
            sim = [sim]
            tmax = [tmax]
        else:
            if tmax is None:
                tmax = [None]*len(sim)
            if len(tmax) != len(sim):
                raise ValueError("When passing multiple simulations, need to pass a list for tmax of the same length (or not pass tmax to default to 10^9 innermost planet orbits for all simulations)")

        args = []
        for i, s in enumerate(sim):
            s = s.copy()
            if match_training == True:
                init_sim_parameters(s)
            else:
                init_sim_parameters(s, megno=False, safe_mode=0)
            minP = np.min([p.P for p in s.particles[1:s.N_real]])
            if tmax[i] is None:
                tmax[i] = 1e9*minP
            if archive_filename:
                if archive_interval is None:
                    archive_interval = tmax[i]/1000
                if len(sim) == 1: # single sim
                    s.automateSimulationArchive(archive_filename, archive_interval, deletefile=True)
                else: # strip final extension and add an index for each simarchive
                    ext = archive_filename.split('.')[-1]
                    len_ext = len(ext)+1
                    filename = "{0}_{1}.{2}".format(archive_filename[:-len_ext], i, ext)
                    s.automateSimulationArchive(filename, archive_interval, deletefile=True)
            args.append([s, tmax[i]])
        
        def run(params):
            sim, tmax = params
            if np.isnan(sim.dt): # sim.dt set to nan for hyperbolic initial conditions in simsetup/set_integrator_and_timestep
                return np.nan
            try:
                sim.integrate(tmax, exact_finish_time=0)
            except (rebound.Collision, rebound.Escape):
                if sim._simulationarchive_filename:
                    sim.simulationarchive_snapshot(sim._simulationarchive_filename.decode('utf-8'))
                return sim.t
           
            return tmax

        if len(args) == 1: # single sim
            tinst = run(args[0])
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            pool = ThreadPool(n_jobs)
            tinst = np.array(pool.map(run, args))
        
        # Uncertainty estimates come from Hussain & Tamayo 2020. Extra factor of sqrt(2) comes from fact that in addition to having a spread in instability times from chaos, we don't know where the mean is from a single integration. See paper
        lower = 10**(np.log10(tinst)-np.sqrt(2)*0.43)
        upper = 10**(np.log10(tinst)+np.sqrt(2)*0.43)

        if len(args) == 1:
            if tinst == tmax[0]:
                lower = 0
                upper = 0
        else: # set confidence intervals for all sims that reached tmax to zero 
            tmax = np.array(tmax)
            lower[tinst == tmax] = 0
            upper[tinst == tmax] = 0 

        return tinst, lower, upper

    def predict_stable(self, sim, tmax=None, archive_filename=None, archive_interval=None, n_jobs=-1, match_training=False):
        """
        Predict whether system is stable up to time=tmax through N-body integration.

        Parameters:

        sim (rebound.Simulation):   Orbital configuration to test
        tmax (float):               Maximum time to integrate for (in Simulation time units) 
        archive_filename (str):     Path and filename to store a rebound SimulationArchive of snapshots
        archive_interval (float):   Time between snapshots for SimulationArchive (in Simulation time units)
        n_jobs (int):               Number of cores to use for calculation (only if passing more than one simulation). Default: Use all available cores. 
        match_training (bool):      In order to match the exact chaotic trajectory given an input orbital configuration from our training set, need to set match_training=True. False gives an equally valid chaotic realization, and gives a factor of ~2 speedup in evaluation.

        Returns:

        float:  Time of instability (Hill sphere crossing), in Simulation time units, or tmax if stable
        int:    1 if integration reached tmax, 0 if Hill spheres crossed
        """
 
        tinst, lower, upper = self.predict_instability_time(sim, tmax, archive_filename, archive_interval, n_jobs, match_training) 
        # If tmax == None, each sim can have different tmaxs
        # Use fact that if confidence intervals == 0, we hit tmax limit
        # (lower = np.nan if error, >0 if hit instability)
        stable = (lower == 0) 
        try:
            return stable.astype(int)
        except:
            return int(stable) # single simulation
