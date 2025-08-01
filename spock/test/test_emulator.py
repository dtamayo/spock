import unittest
import rebound as rb
from spock import GiantImpactPhaseEmulator
import numpy as np

def mayasim():
    sim = rb.Simulation()
    sim.G = 4*np.pi**2
    sim.add(m=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
    sim.add(m=3e-06, x=-0.04800539395878285, y=-0.014405856984195031, z=-0.0003664160546367002, vx=7.748799320461535, vy=-26.938089544570456, vz=0.058605713117226184)
    sim.add(m=3e-06, x=-0.03249737600195156, y=-0.03879749442803396, z=-0.0002652168123928787, vx=21.606681145306528, vy=-17.84084193326855, vz=-0.10788675140543141)
    sim.add(m=3e-06, x=0.04652083205368205, y=-0.022350676162887412, z=-0.00011874925458699267, vx=11.952938650881512, vy=25.02425563180238, vz=-0.4013245131467435)
    sim.add(m=3e-06, x=0.013725882137906861, y=-0.05168492115067017, z=-0.0006686437687299294, vx=26.15780389536077, vy=6.758414107037664, vz=-0.0751699060113149)
    sim.add(m=3e-06, x=-0.01299415729373822, y=-0.05308147731744508, z=0.0003920614467076649, vx=26.000707610364497, vy=-6.006567638025953, vz=-0.2430243223817451)
    sim.add(m=3e-06, x=-0.045483229353683485, y=-0.030188819312691687, z=-2.612973746328278e-05, vx=14.850799637900632, vy=-22.509645396422506, vz=0.35016084068310177)
    sim.add(m=3e-06, x=-0.05315394126216153, y=0.019155795279835262, z=0.00016009169679976847, vx=-9.404974084271403, vy=-24.56232307370846, vz=-0.08463252642819143)
    sim.add(m=3e-06, x=0.03304211596842195, y=-0.046631352244268225, z=-0.0002572402627276085, vx=21.550614516229874, vy=14.982661728893587, vz=0.195239513079604)
    return sim

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
        with self.assertRaises(rb.ParticleNotFound):
            sim = self.model.predict(sim)
            p = sim.particles['hyperbolic']
    
    def test_escaper(self):
        sim = escapesim()
        with self.assertRaises(rb.ParticleNotFound):
            sim = self.model.predict(sim)
            p = sim.particles['escaper']

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
        self.model = GiantImpactPhaseEmulator(seed=0)
        sim = unstablesim()
        L0 = sim.angular_momentum()
        pred_sim = self.model.predict(sim)
        L = pred_sim.angular_momentum()
        for i in range(3):
            self.assertAlmostEqual(L0[i], L[i], delta=0.1*L0[2]) # must agree to within 10% of initial Lz value
            
    def test_E_conservation(self):
        self.model = GiantImpactPhaseEmulator(seed=0)
        sim = unstablesim()
        E0 = sim.energy()
        pred_sim = self.model.predict(sim)

        E = pred_sim.energy()
        self.assertAlmostEqual(E0, E, delta=0.25*abs(E0)) # must agree to within 25% of initial E value

    def test_seed(self):
        xs = []
        for i in range(5):
            model = GiantImpactPhaseEmulator(seed=0)
            sim = mayasim()
            sim = model.step(sim, tmaxs=1e7)
            xs.append(sim.particles[2].x)
        self.assertTrue(all(x==xs[0] for x in xs))

    def test_step_equivalence(self):
        sim = unstablesim()
        model = GiantImpactPhaseEmulator(seed=0)
        sim1 = model.predict(sim)
       
        # should give the same as running a sequence of steps
        sim2 = unstablesim()
        tmax = 1e9*sim2.particles[1].P
        model = GiantImpactPhaseEmulator(seed=0)
        for i in range(3):
            sim2 = model.step(sim2, tmaxs=tmax)
        self.assertAlmostEqual(sim1.particles[1].P, sim2.particles[1].P, delta=1.e-10)

if __name__ == '__main__':
    unittest.main()
