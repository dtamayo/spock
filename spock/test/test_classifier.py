import rebound
import unittest
from spock import StabilityClassifier
from spock.simsetup import init_sim_parameters

def rescale(sim, dscale, tscale, mscale):                                                                      
    simr = rebound.Simulation()
    vscale = dscale/tscale 
    simr.G *= mscale*tscale**2/dscale**3

    for p in sim.particles:
        simr.add(m=p.m/mscale, x=p.x/dscale, y=p.y/dscale, z=p.z/dscale, vx=p.vx/vscale, vy=p.vy/vscale, vz=p.vz/vscale, r=p.r/dscale)

    return simr

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.model = StabilityClassifier()

    def test_singlelisttrios(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            self.model.predict_stable(sim, trios=[1,2,3])
    
    def test_wrongindex(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            self.model.predict_stable(sim, trios=[[1,2,4]])
    
    def test_zeroindex(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            self.model.predict_stable(sim, trios=[[0,2,4]])
    
    def test_wronglengthtrios(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            self.model.predict_stable(sim, trios=[[1,2,3], [1,2]])
    
    def test_indexorderinvariance(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p123 = self.model.predict_stable(sim, trios=[[1,2,3]])
        p231 = self.model.predict_stable(sim, trios=[[2,3,1]])
        self.assertEqual(p123, p231)
    
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
        sim = rebound.Simulation('longstable.bin')
        init_sim_parameters(sim)
        _, _ = self.model.generate_features(sim, copysim=False)
        sim.integrate(2.e4, exact_finish_time=0)
        x1 = sim.particles[1].x

        # CHANGE TO USING NBODY
        sim = rebound.Simulation('longstable.bin')
        init_sim_parameters(sim)
        sim.integrate(2.e4, exact_finish_time=0)
        x2 = sim.particles[1].x
        self.assertEqual(x1, x2)
   
    # when chaotic realization matters, probs will vary more (eg t_inst=2e4)
    def test_galilean_transformation(self):
        sim = rebound.Simulation('longstable.bin')
        sim.move_to_com()
        p_com = self.model.predict_stable(sim)

        sim = rebound.Simulation('longstable.bin')
        for p in sim.particles:
            p.vx += 1000
        p_moving = self.model.predict_stable(sim)
        self.assertAlmostEqual(p_com, p_moving, delta=1.e-4)
   
    def test_rescale_distances(self):
        sim = rebound.Simulation('longstable.bin')
        p0 = self.model.predict_stable(sim)

        sim = rebound.Simulation('longstable.bin')
        sim = rescale(sim, dscale=1e10, tscale=1, mscale=1)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-4)
    
    def test_rescale_times(self):
        sim = rebound.Simulation('longstable.bin')
        p0 = self.model.predict_stable(sim)

        sim = rebound.Simulation('longstable.bin')
        sim = rescale(sim, dscale=1, tscale=1e10, mscale=1)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-4)

    def test_rescale_masses(self):
        sim = rebound.Simulation('longstable.bin')
        p0 = self.model.predict_stable(sim)

        sim = rebound.Simulation('longstable.bin')
        sim = rescale(sim, dscale=1, tscale=1, mscale=1e10)
        p1 = self.model.predict_stable(sim)
        self.assertAlmostEqual(p0, p1, delta=1.e-4)
    
    def test_hyperbolic(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, a=-1., e=1.2)
        sim.add(m=1.e-5, a=2.)
        sim.add(m=1.e-5, a=3.)
        self.assertEqual(self.model.predict_stable(sim), 0)
    
    def test_unstable_in_short_integration(self):
        sim = rebound.Simulation('unstable.bin')
        self.assertEqual(self.model.predict_stable(sim), 0)
    
    def test_solarsystem(self):
        sim = rebound.Simulation('solarsystem.bin')
        self.assertGreater(self.model.predict_stable(sim), 0.7)
    
    def test_stable(self):
        sim = rebound.Simulation('longstable.bin')
        self.assertGreater(self.model.predict_stable(sim), 0.7)
    
if __name__ == '__main__':
    unittest.main()
