import numpy as np
import rebound
from .simsetup import init_sim_parameters

class NbodyRegressor():
    def __init__(self):
        pass 
    
    def predict_instability_time(self, sim, tmax=None, archive_filename=None, archive_interval=None):
        """
        Predict instability time through N-body integration.

        Parameters:

        sim (rebound.Simulation):   Orbital configuration to test
        tmax (float):               Maximum time to integrate for (in Simulation time units) 
        archive_filename (str):     Path and filename to store a rebound SimulationArchive of snapshots
        archive_interval (float):   Time between snapshots for SimulationArchive (in Simulation time units)

        Returns:

        float:  Time of instability (Hill sphere crossing), in Simulation time units, or tmax if stable
        int:    1 if integration reached tmax, 0 if Hill spheres crossed
        """
        sim = sim.copy()
        init_sim_parameters(sim)

        minP = np.min([p.P for p in sim.particles[1:sim.N_real]])
        if tmax is None:
            tmax = 1e9*minP
        if archive_filename:
            if archive_interval is None:
                archive_interval = tmax/1000
            sim.automateSimulationArchive(archive_filename, archive_interval, deletefile=True)

        try:
            sim.integrate(tmax, exact_finish_time=0)
        except rebound.Collision:
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return sim.t
           
        return tmax

    def predict_stable(self, sim, tmax=None, archive_filename=None, archive_interval=None):
        minP = np.min([p.P for p in sim.particles[1:sim.N_real]])
        if tmax is None:
            tmax = 1e9*minP
        tinst = self.predict_instability_time(sim, tmax, archive_filename, archive_interval) 
        
        if tinst >= tmax: # always check for stability. sim.dt and tinst will be nan if hyperbolic
            return 1
        else:
            return 0
        
