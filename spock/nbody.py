from .simsetup import init_sim_parameters, check_valid_sim, check_hyperbolic

class Nbody():
    def __init__(self:  # don't allow reloading simarcchives. give way for them to do it  themselves
        pass 
    def predict_stable(sim, dtfrac=0.05, tmax=None, archive_filename=None, archive_interval=None): 
       
        check_valid_sim(sim)
        sim = init_sim_parameters(sim, dtfrac)

        minP = np.min([p.P for p in sim.particles[1:]])
        if tmax is None:
            tmax = 1e9*self.sim.particles[1].P
          
        try:
            sim.integrate(tmax, exact_finish_time=0)
            collision=False
        except rebound.Collision:
            collision=True
            if archive_filename:
                sim.simulationarchive_snapshot(archive_filename)
        if collision == True or check_hyperbolic(sim) == True:
            return 0

        return 1
    
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
