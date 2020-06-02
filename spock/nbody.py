import numpy as np
from .simsetup import init_sim_parameters

class Nbody():
    def __init__(self):
        pass 

    def predict_stable(self, sim, tmax=None, dtfrac=0.05, archive_filename=None, archive_interval=None): 
        t_inst = self.predict_instability_time(sim, tmax, dtfrac, archive_filename, archive_interval)

        if t_inst < tmax:
            return 0
        else:
            return 1
    
    def predict_instability_time(self, sim, tmax=None, dtfrac=0.05, archive_filename=None, archive_interval=None):
        sim = sim.copy()
        init_sim_parameters(sim, dtfrac)

        minP = np.min([p.P for p in sim.particles[1:]])
        if tmax is None:
            tmax = 1e9*minP
          
        if archive_filename:
            if archive_interval is None:
                archive_interval = 1e6*minP
            sim.automateSimulationArchive(archive_filename, archive_interval, deletefile=True)

        try:
            sim.integrate(tmax, exact_finish_time=0)
        except rebound.Collision:
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return sim.t 

        return tmax
