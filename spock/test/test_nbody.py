import rebound
import unittest
from spock import Nbody
from spock.simsetup import init_sim_parameters

def rescale(sim, dscale, tscale, mscale):                                                                      
    simr = rebound.Simulation()
    vscale = dscale/tscale 
    simr.G *= mscale*tscale**2/dscale**3

    for p in sim.particles:
        simr.add(m=p.m/mscale, x=p.x/dscale, y=p.y/dscale, z=p.z/dscale, vx=p.vx/vscale, vy=p.vy/vscale, vz=p.vz/vscale, r=p.r/dscale)

    return simr

class TestNbody(unittest.TestCase):
    def setUp(self):
        self.model = Nbody()

    def test_repeat(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p1 = self.model.predict_stable(sim)
        p2 = self.model.predict_stable(sim)
        self.assertEqual(p1, p2)
    
    def test_hyperbolic(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, a=-1., e=1.2)
        sim.add(m=1.e-5, a=2.)
        sim.add(m=1.e-5, a=3.)
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 0)
    
    def test_unstable_in_short_integration(self):
        sim = rebound.Simulation('unstable.bin')
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 0)
    
    def test_solarsystem(self):
        sim = rebound.Simulation('solarsystem.bin')
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 1)
    
    def test_stable(self):
        sim = rebound.Simulation('longstable.bin')
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 1)
    
if __name__ == '__main__':
    unittest.main()
