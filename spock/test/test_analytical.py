import unittest
import numpy as np
import rebound

from spock import AnalyticalClassifier, NbodyRegressor
from spock.feature_functions import get_tseries
from spock.simsetup import setup_sim 


def unstablesimecc():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1, e=0.1, pomega=3)
    sim.add(m=1.e-4, P=1.3, e=0.1, pomega=1)
    sim.add(m=1.e-4, P=1.6, e=0.1, pomega=5)
    for p in sim.particles[1:]:
        p.r = p.a*(p.m/3)**(1/3)
    sim.move_to_com()
    sim.collision='line'
    sim.integrator="whfast"
    sim.dt = 0.05
    return sim

def longstablesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-7, P=1)
    sim.add(m=1.e-7, P=2.1)
    sim.add(m=1.e-7, P=4.5)
    for p in sim.particles[1:]:
        p.r = p.a*(p.m/3)**(1/3)
    sim.move_to_com()
    sim.collision='line'
    sim.integrator="whfast"
    sim.dt = 0.05
    return sim

def hyperbolicsim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, a=-1., e=1.2)
    sim.add(m=1.e-5, a=2.)
    sim.add(m=1.e-5, a=3.)
    return sim

def unstable2psim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1)
    sim.add(m=1.e-4, P=1.01, f=np.pi)
    return sim

def unstable2psimhyperbolic():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, a=-1, e=1.01)
    sim.add(m=1.e-4, a=-1.05, e=1.01, f=np.pi/6)
    return sim

def unstable2psimhighe():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, a=1, e=.99)
    sim.add(m=1.e-4, a=1.05, e=0.99, f=np.pi/6)
    return sim

def stable2psim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1)
    sim.add(m=1.e-4, P=2.3, f=np.pi)
    return sim

def singlesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1)
    return sim

def rescale(sim, dscale, tscale, mscale):                                                                      
    simr = rebound.Simulation()
    vscale = dscale/tscale 
    simr.G *= mscale*tscale**2/dscale**3

    for p in sim.particles:
        simr.add(m=p.m/mscale, x=p.x/dscale, y=p.y/dscale, z=p.z/dscale, vx=p.vx/vscale, vy=p.vy/vscale, vz=p.vz/vscale, r=p.r/dscale)

    return simr

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.model = AnalyticalClassifier()
    
    def test_list(self):
        stable = self.model.predict_stable([hyperbolicsim(), unstablesimecc(), longstablesim()])
        self.assertEqual(stable[0], 0)
        self.assertEqual(stable[1], 0)
        self.assertGreater(stable[2], 0.7) 
    
    def test_sim_unchanged(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        sim.integrate(1.2)
        x0 = sim.particles[1].x
        p1 = self.model.predict_stable(sim)
        self.assertEqual(sim.particles[1].x, x0)

    def test_repeat(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p1 = self.model.predict_stable(sim)
        p2 = self.model.predict_stable(sim)
        self.assertEqual(p1, p2)
    
    def test_same_trajectory(self):
        sim = longstablesim()
        sim = setup_sim(sim)
        _, _ = get_tseries(sim, (1e4, 80, [[1,2,3]]))
        x1 = sim.particles[1].x

        sim = longstablesim()
        nbody = NbodyRegressor()
        nbody.predict_stable(sim, tmax=1e4, archive_filename='temp.bin', archive_interval=1.e4)
        sa = rebound.Simulationarchive('temp.bin')
        sim = sa[-1]
        x2 = sim.particles[1].x
        self.assertAlmostEqual(x1, x2, delta=1.e-5)
   
    def test_single(self):
        sim = singlesim()
        prob = self.model.predict_stable(sim)
        self.assertEqual(prob, 1)

    def test_single_list(self):
        sims = [singlesim(), singlesim()]
        probs = self.model.predict_stable(sims)
        self.assertTrue(all(prob == 1 for prob in probs))

    def test_stable2p(self):
        sim = stable2psim()
        prob = self.model.predict_stable(sim)
        self.assertEqual(prob, 1)

    def test_unstable2p(self):
        sim = unstable2psim()
        prob = self.model.predict_stable(sim)
        self.assertEqual(prob, 0)

    def test_unstable2phyperbolic(self):
        sim = unstable2psimhyperbolic()
        prob = self.model.predict_stable(sim)
        self.assertEqual(prob, 0)

    def test_unstable2phighe(self):
        sim = unstable2psimhighe()
        prob = self.model.predict_stable(sim)
        self.assertEqual(prob, 0)

    def test_stable2p_list(self):
        sims = [stable2psim(), stable2psim()]
        probs = self.model.predict_stable(sims)
        self.assertTrue(all(prob == 1 for prob in probs))
 
    def test_unstable2p_list(self):
        sims = [unstable2psim(), unstable2psim()]
        probs = self.model.predict_stable(sims)
        self.assertTrue(all(prob == 0 for prob in probs))
    
    # when chaotic realization matters, probs will vary more (eg t_inst=2e4)
    def test_galilean_transformation(self):
        sim = longstablesim()
        sim.move_to_com()
        p_com = self.model.predict_stable(sim)

        sim = longstablesim()
        for p in sim.particles:
            p.vx += 1000
        p_moving = self.model.predict_stable(sim)
        self.assertAlmostEqual(p_com, p_moving, delta=1.e-2)
    
    def test_rescale_distances(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim)

        sim = longstablesim()
        sim = rescale(sim, dscale=1e10, tscale=1, mscale=1)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-2)
    
    def test_rescale_times(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim)

        sim = longstablesim()
        sim = rescale(sim, dscale=1, tscale=1e10, mscale=1)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-1)
    
    def test_rescale_masses(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim)

        sim = longstablesim()
        sim = rescale(sim, dscale=1, tscale=1, mscale=1e10)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-2)
    
    def test_stable(self):
        sim = longstablesim()
        self.assertGreater(self.model.predict_stable(sim), 0.7)
    
    
    def test_unstable_in_short_integration(self):
        sim = unstablesimecc()
        self.assertEqual(self.model.predict_stable(sim), 0)
   
    def test_hyperbolic(self):
        sim = hyperbolicsim()
        self.assertEqual(self.model.predict_stable(sim), 0)

if __name__ == '__main__':
    unittest.main()
