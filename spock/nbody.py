import numpy as np
import rebound
from .simsetup import init_sim_parameters

class Nbody():
    def __init__(self):
        pass 

    def predict_stable(self, sim, tmax=None, archive_filename=None, archive_interval=None): 
        t_inst = self.predict_instability_time(sim, tmax, archive_filename, archive_interval)
        
        minP = np.min([p.P for p in sim.particles[1:sim.N_real]])
        if t_inst >= 1e9*minP:
            return 1
        else:
            return 0
    
    def predict_instability_time(self, sim, tmax=None, archive_filename=None, archive_interval=None):
        sim = sim.copy()
        init_sim_parameters(sim)

        minP = np.min([p.P for p in sim.particles[1:sim.N_real]])
        if tmax is None:
            tmax = 1e9*minP
        if archive_filename:
            if archive_interval is None:
                archive_interval = 1e6*minP
            sim.automateSimulationArchive(archive_filename, archive_interval, deletefile=True)

        try:
            sim.integrate(tmax, exact_finish_time=0)
        except:# rebound.Collision:
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return sim.t
            
        return tmax
