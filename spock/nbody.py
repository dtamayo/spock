from .simsetup import init_sim

class Nbody():
    def __init__(self:  # don't allow reloading simarcchives. give way for them to do it  themselves
        pass 
    def predict_stable(sim, tmax=None, archive_filename=None, interval=None): # WILL ALWAYS OVERWRITE. WE"re  NOT PROVIDING OPTION TO RELOAD IN   MIDDL
        
        sim = init_sim(sim, dtfrac, archive_filename, interval)
        if tmax is None:
            tmax = 1e9*self.sim.particles[1].P
          

        try:
            sim.integrate(tmax, exact_finish_time=0)
        except rebound.Collision:
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return False

        return True
    
    def predict_instability_time(sim, tmax=None, dtfrac=0.05, archive_filename=None, interval=None): 
        sim = init_sim(sim, dtfrac, archive_filename, interval)
        if tmax is None:
            tmax = 1e9*self.sim.particles[1].P

        try:
            sim.integrate(tmax, exact_finish_time=0)
        except rebound.Collision:
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
            return sim.t 

        return tmax
