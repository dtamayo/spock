import rebound
import unittest
import numpy as np
from spock import StabilityClassifier
from spock.simsetup import init_sim_parameters

class TestSimSetup(unittest.TestCase):
    def setUp(self):
        self.model = StabilityClassifier()

    def test_negmass(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=-1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            init_sim_parameters(sim)
        
    def test_descending_periods(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=3.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=1.)
        init_sim_parameters(sim)
        self.assertAlmostEqual(sim.dt, 0.05, delta=1e-15)
    
    def test_extreme_ecc(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1., e=0.999)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        init_sim_parameters(sim)
        self.assertEqual(sim.integrator, 'ias15')
    
    def test_high_ecc(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1., e=0.9)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        init_sim_parameters(sim)
        self.assertEqual(sim.integrator, 'whfast')
    
    def test_second_p_ecc(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=10., e=0.99)
        sim.add(m=1.e-5, P=3.)
        init_sim_parameters(sim)
        self.assertAlmostEqual(sim.dt, 0.05*10*0.01**1.5/np.sqrt(1.99), delta=1.e-8)
    
    def test_set_collision(self):
        sim = rebound.Simulation('unstable.bin')
        init_sim_parameters(sim)
        try:
            sim.integrate(1e4*sim.particles[1].P)
        except:
            pass
        self.assertEqual(sim._status, 5)

if __name__ == '__main__':
    unittest.main()
