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
        return self._predict_instability_time_nocopy(sim, tmax, archive_filename, archive_interval) 

    def sample_instability_time(self, sim, tmax=None, archive_filename=None, archive_interval=None):
        """
        Sample instability time through N-body integration, by applying small random offset to
        initial conditions. See Hussain & Tamayo (2020).

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
        delta = 1.e-12
        sim.particles[1].x *= 1.+delta*np.random.random()
        return self._predict_instability_time_nocopy(sim, tmax, archive_filename, archive_interval) 

    def _predict_instability_time_nocopy(self, sim, tmax=None, archive_filename=None, archive_interval=None):
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
            stable = 0
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return sim.t, stable
           
        stable = 1
        return tmax, stable

        



