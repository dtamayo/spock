import rebound
import unittest
from spock import NbodyRegressor, DeepRegressor
import numpy as np

SAMPLE_SETTINGS = dict(samples=1000, max_model_samples=30)

def unstablesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1)
    sim.add(m=1.e-4, P=1.3)
    sim.add(m=1.e-4, P=1.6)
    return sim

def longstablesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-7, P=1)
    sim.add(m=1.e-7, P=2.1)
    sim.add(m=1.e-7, P=4.5)
    return sim

def solarsystemsim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.7e-7, a=0.39, e=0.21)
    sim.add(m=2.4e-6, a=0.72, e=0.007)
    sim.add(m=3.e-6, a=1, e=0.017)
    sim.add(m=3.2e-7, a=1.52, e=0.09)
    sim.add(m=1.e-3, a=5.2, e=0.049)
    sim.add(m=2.9e-4, a=9.54, e=0.055)
    sim.add(m=4.4e-5, a=19.2, e=0.047)
    sim.add(m=5.2e-5, a=30.1, e=0.009)
    return sim

def hyperbolicsim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, a=-1., e=1.2)
    sim.add(m=1.e-5, a=2.)
    sim.add(m=1.e-5, a=3.)
    return sim

def escapesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-12, P=3.14, e=0.03, l=0.5)
    sim.add(m=1.e-12, P=4.396, e=0.03, l=4.8)
    sim.add(m=1.e-12, a=100, e=0.999)
    return sim

def vstablesim():
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=1e-7, P=1.)
    sim.add(m=1e-7, P=1.8)
    sim.add(m=1e-7, P=3.2)
    return sim

def rescale(sim, dscale, tscale, mscale):                                                                      
    simr = rebound.Simulation()
    vscale = dscale/tscale 
    simr.G *= mscale*tscale**2/dscale**3

    for p in sim.particles:
        simr.add(m=p.m/mscale, x=p.x/dscale, y=p.y/dscale, z=p.z/dscale, vx=p.vx/vscale, vy=p.vy/vscale, vz=p.vz/vscale, r=p.r/dscale)

    return simr

class TestRegressor(unittest.TestCase):
    def setUp(self):
        self.model = DeepRegressor(cuda=False)

    def relative(self, p1, p2):
        return abs(p1 - p2) / (p1 + p2) / 2

    def test_seed(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p1 = np.log10(self.model.predict_instability_time(sim, seed=0, **SAMPLE_SETTINGS)[0])
        p2 = np.log10(self.model.predict_instability_time(sim, seed=1, **SAMPLE_SETTINGS)[0])
        self.assertTrue(self.relative(p1, p2) < 0.1)
        self.assertTrue(self.relative(p1, p2) > 0.0)

   
    def test_prediction(self):
        times = []
        sims = []
        for mass in [1e-4, 5e-5, 3e-5, 1e-5]:
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=mass, P=1)
            sim.add(m=mass, P=1.3)
            sim.add(m=mass, P=1.6)
            sims.append(sim)

        times = np.log10(self.model.predict_instability_time(sims, **SAMPLE_SETTINGS)[0])
        # First one is unstable:
        self.assertTrue(times[0] < 4.0)
        # Should get more stable:
        self.assertTrue(np.all(times[1:] > times[:-1]))
   
    def test_rescale_distances(self):
        sim = longstablesim()
        t, upper, lower = self.model.predict_instability_time(sim, seed=0, **SAMPLE_SETTINGS)

        simr = rescale(sim, dscale=1e10, tscale=1, mscale=1)
        tr, upperr, lowerr = self.model.predict_instability_time(simr, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(t/sim.particles[1].P, tr/simr.particles[1].P, delta=np.abs((upper-lower)/10/sim.particles[1].P))
    
    def test_rescale_times(self):
        sim = longstablesim()
        t, upper, lower = self.model.predict_instability_time(sim, seed=0, **SAMPLE_SETTINGS)

        simr = rescale(sim, dscale=1, tscale=1e10, mscale=1)
        tr, upperr, lowerr = self.model.predict_instability_time(simr, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(t/sim.particles[1].P, tr/simr.particles[1].P, delta=np.abs((upper-lower)/10/sim.particles[1].P))

    def test_rescale_masses(self):
        sim = longstablesim()
        t, upper, lower = self.model.predict_instability_time(sim, seed=0, **SAMPLE_SETTINGS)

        simr = rescale(sim, dscale=1, tscale=1, mscale=1e10)
        tr, upperr, lowerr = self.model.predict_instability_time(simr, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(t/sim.particles[1].P, tr/simr.particles[1].P, delta=np.abs((upper-lower)/10/sim.particles[1].P))

    def test_time_scaling(self):
        times = []
        sims = []
        mass = 3e-5
        for P in [1, 10]:
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=mass, P=1*P)
            sim.add(m=mass, P=1.3*P)
            sim.add(m=mass, P=1.6*P)
            sims.append(sim)

        # Second time should have ~10x larger inst time.
        times = self.model.predict_instability_time(sims, seed=0, **SAMPLE_SETTINGS)[0]
        # Should be much larger time:
        self.assertGreater(times[1], 5*times[0])

    def test_time_scaling_from_integration(self):
        times = []
        sims = []
        mass = 1e-3
        for P in [1, 10]:
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=mass, P=1*P)
            sim.add(m=mass, P=1.3*P)
            sim.add(m=mass, P=1.6*P)
            sims.append(sim)

        # Second time should have ~10x larger inst time.
        times = self.model.predict_instability_time(sims, seed=0, **SAMPLE_SETTINGS)[0]
        # Should be much larger time:
        self.assertGreater(times[1], 5*times[0])

    def test_custom_prior(self):
        mass = 1e-7

        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=mass, P=1)
        sim.add(m=mass, P=1.3)
        sim.add(m=mass, P=1.6)
        expected_center = 13.0
        prior = lambda logT: np.exp(-(logT - expected_center)**2/2/0.1**2)

        times = np.log10(self.model.predict_instability_time(sim, prior_above_9=prior, **SAMPLE_SETTINGS)[0])
        self.assertAlmostEqual(times, expected_center, delta=1e-1)
    
    def test_list_time(self):
        tinst, lower, upper = self.model.predict_instability_time([hyperbolicsim(), escapesim(), unstablesim(), longstablesim()])
        self.assertTrue(np.isnan(tinst[0]))
        self.assertLess(tinst[1], 1e4)
        self.assertLess(tinst[2], 1e4)
        self.assertGreater(tinst[3], 1e4)

class TestRegressorClassification(unittest.TestCase):
    def setUp(self):
        self.model = DeepRegressor(cuda=False)
    
    def test_list_stable(self): # pass list of sims with same size list of tmax
        tmax = [1e4, 1e4, 1, 1e4] # test that unstablesim in middle still classified as stable with tmax=1
        stable_target = [0, 0, 1, 1]
        stable = self.model.predict_stable([hyperbolicsim(), escapesim(), unstablesim(), longstablesim()], tmax=tmax)
        self.assertSequenceEqual(stable.tolist(), stable_target)
    
    def test_list_no_tmax(self): # pass list of sims, tmax = None
        stable = self.model.predict_stable([vstablesim(), vstablesim()])
        self.assertGreater(stable[0], 0.9)
        self.assertGreater(stable[1], 0.9)
    
    def test_single_no_tmax(self): # pass list of sims, tmax = None
        stable = self.model.predict_stable(vstablesim())
        self.assertGreater(stable, 0.9)
    
    def test_mismatched_lists(self):
        tmax = [1e4, 1e4, 1] # test that unstablesim in middle still classified as stable with tmax=1
        with self.assertRaises(AssertionError):
            stable = self.model.predict_stable([hyperbolicsim(), escapesim(), unstablesim(), longstablesim()], tmax=tmax)

    def test_sim_unchanged(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        sim.integrate(1.2)
        x0 = sim.particles[1].x
        p1 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertEqual(sim.particles[1].x, x0)

    def test_repeat(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p1 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        p2 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertEqual(p1, p2)
   
    # when chaotic realization matters, probs will vary more (eg t_inst=2e4)
    def test_galilean_transformation(self):
        sim = longstablesim()
        sim.move_to_com()
        p_com = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)

        sim = longstablesim()
        for p in sim.particles:
            p.vx += 1000
        p_moving = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(p_com, p_moving, delta=1e-2)
   
    def test_rescale_distances(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)

        sim = longstablesim()
        sim = rescale(sim, dscale=1e10, tscale=1, mscale=1)
        p1 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(p0, p1, delta=1e-2)
    
    def test_rescale_times(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)

        sim = longstablesim()
        sim = rescale(sim, dscale=1, tscale=1e10, mscale=1)
        p1 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(p0, p1, delta=1e-2)

    def test_rescale_masses(self):
        sim = longstablesim()
        p0 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)

        sim = longstablesim()
        sim = rescale(sim, dscale=1, tscale=1, mscale=1e10)
        p1 = self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS)
        self.assertAlmostEqual(p0, p1, delta=1e-2)
    
    def test_hyperbolic(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, a=-1., e=1.2)
        sim.add(m=1.e-5, a=2.)
        sim.add(m=1.e-5, a=3.)
        self.assertEqual(self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS), 0)
    
    def test_escape(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-12, P=3.14, e=0.03, l=0.5)
        sim.add(m=1.e-12, P=4.396, e=0.03, l=4.8)
        sim.add(m=1.e-12, a=100, e=0.999)
        self.assertEqual(self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS), 0)
    
    def test_unstable_in_short_integration(self):
        sim = unstablesim()
        self.assertEqual(self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS), 0)
    
    def test_solarsystem(self):
        sim = solarsystemsim()
        median, lower, upper, t_inst_samples = self.model.predict_instability_time(
                sim, seed=0,
                return_samples=True,
                **SAMPLE_SETTINGS)
        log_iqr = np.log10(upper) - np.log10(lower)
        correct_estimate = (np.average(t_inst_samples > 1e9) > 0.7 )
        very_uncertain = (log_iqr > 3)
        self.assertTrue(correct_estimate or very_uncertain)

    def test_stable(self):
        sim = longstablesim()
        self.assertGreater(self.model.predict_stable(sim, seed=0, **SAMPLE_SETTINGS), 0.7)

if __name__ == '__main__':
    unittest.main()
