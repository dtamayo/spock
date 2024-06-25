import unittest
import rebound as rb
from spock import GiantImpactPhaseEmulator

def unstablesim():
    sim = rb.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, P=1, e=0.03, pomega=2., l=0.5)
    sim.add(m=1.e-5, P=1.2, e=0.03, pomega=3., l=3.)
    sim.add(m=1.e-5, P=1.5, e=0.03, pomega=1.5, l=2.)
    sim.add(m=1.e-5, P=2.2, e=0.03, pomega=0.5, l=5.0)
    sim.add(m=1.e-5, P=2.5, e=0.03, pomega=5.0, l=1.5)
    return sim

def largePsim():
    sim = rb.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, P=10, e=0.03, pomega=2., l=0.5)
    sim.add(m=1.e-5, P=12, e=0.03, pomega=3., l=3.)
    sim.add(m=1.e-5, P=15, e=0.03, pomega=1.5, l=2.)
    sim.add(m=1.e-5, P=22, e=0.03, pomega=0.5, l=5.0)
    sim.add(m=1.e-5, P=25, e=0.03, pomega=5.0, l=1.5)
    return sim

def stablesim():
    sim = rb.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, P=1.)
    sim.add(m=1.e-5, P=2.)
    sim.add(m=1.e-5, P=3.)
    return sim

def hyperbolicsim():
    sim = rb.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, a=-1., e=1.2, hash='hyperbolic')
    sim.add(m=1.e-5, a=2.)
    sim.add(m=1.e-5, a=3.)
    return sim

def escapesim():
    sim = rb.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-12, P=3.14, e=0.03, l=0.5)
    sim.add(m=1.e-12, P=4.396, e=0.03, l=4.8)
    sim.add(m=1.e-12, a=100, e=0.999, hash='escaper')
    return sim

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.model = GiantImpactPhaseEmulator(seed=0)
   
    def test_hyperbolic(self):
        sim = hyperbolicsim()
        pred_sim = self.model.predict(sim)
        self.assertRaises(rb.ParticleNotFound, pred_sim.particles['hyperbolic'])
    
    def test_escaper(self):
        sim = escapesim()
        pred_sim = self.model.predict(sim)
        self.assertRaises(rb.ParticleNotFound, pred_sim.particles['escaper'])
               
    def test_stable(self):
        sim = stablesim()
        N = sim.N
        pred_sim = self.model.predict(sim)
        self.assertEqual(pred_sim.N, N)
    
    def test_unstable(self):
        sim = unstablesim()
        N = sim.N
        pred_sim = self.model.predict(sim)
        self.assertLess(pred_sim.N, N)

    def test_scale_invariance(self):
        sim = largePsim()
        Pmin = sim.particles[1].P
        pred_sim = self.model.predict(sim)
        self.assertGreater(pred_sim.particles[1].P, 0.99*Pmin)

    def test_multiple(self):
        sims = [stablesim(), unstablesim()]
        sims = self.model.predict(sims) # just test there are no exceptions

    def test_default_tmaxs(self):
        sim = unstablesim()
        model = GiantImpactPhaseEmulator(seed=0)
        pred_sim = model.predict(sim, tmaxs=1e9*sim.particles[1].P)
        sim = unstablesim()
        model = GiantImpactPhaseEmulator(seed=0)
        pred_sim2 = model.predict(sim)
        self.assertAlmostEqual(pred_sim.particles[1].P, pred_sim2.particles[1].P, delta=1.e-10)

    def test_L_conservation(self):
        sim = unstablesim()
        L0 = sim.angular_momentum()
        pred_sim = self.model.predict(sim)
        L = pred_sim.angular_momentum()
        for i in range(3):
            self.assertAlmostEqual(L0[i], L[i], delta=0.1*L0[2]) # must agree to within 10% of initial Lz value
            
    def test_E_conservation(self):
        sim = unstablesim()
        E0 = sim.energy()
        sim = self.model.predict(sim)
        E = sim.angular_momentum()
        self.assertAlmostEqual(E0, E, delta=0.25*E0) # must agree to within 25% of initial E value

    def test_step_equivalence(self):
        sim = unstablesim()
        tmax = 1e9*sim.particles[1].P # cache since this will change if we take multiple steps and inner planet merges
        model = GiantImpactPhaseEmulator(seed=0)
        pred_sim = model.predict(sim, tmaxs=tmax)
        model = GiantImpactPhaseEmulator(seed=0)
        sim2 = unstablesim()
        tmax = 1e9*sim2.particles[1].P # cache since this will change if we take multiple steps and inner planet merges
        for i in range(3):
            pred_sim2 = model.step(sim2, tmaxs=tmax)
        self.assertAlmostEqual(pred_sim.particles[1].P, pred_sim2.particles[1].P, delta=1.e-10)

if __name__ == '__main__':
    unittest.main()
