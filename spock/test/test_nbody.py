import numpy as np
import unittest
import rebound
from spock import NbodyRegressor
from spock.simsetup import init_sim_parameters
from spock.feature_functions import features

def unstablesim():
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-4, P=1)
    sim.add(m=1.e-4, P=1.3)
    sim.add(m=1.e-4, P=1.6)
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

class TestNbody(unittest.TestCase):
    def setUp(self):
        self.model = NbodyRegressor()

    def test_list_stable(self):
        tmax = [1e4, 1e4, 1, 1e4, 1e4] # test that unstablesim in middle still classified as stable with tmax=1
        stable_target = [0, 0, 1, 1, 1]
        stable = self.model.predict_stable([hyperbolicsim(), escapesim(), unstablesim(), longstablesim(), solarsystemsim()], tmax=tmax)
        self.assertSequenceEqual(stable.tolist(), stable_target)
    
    def test_mismatched_lists(self):
        tmax = [1e4, 1e4, 1] # test that unstablesim in middle still classified as stable with tmax=1
        with self.assertRaises(ValueError):
            stable = self.model.predict_stable([hyperbolicsim(), escapesim(), unstablesim(), longstablesim()], tmax=tmax)

    def test_list_time(self):
        tmax = [1e4, 1e4, 1, 1e4, 1e4] # test that unstablesim in middle still classified as stable with tmax=1
        tinst, lower, upper = self.model.predict_instability_time([hyperbolicsim(), escapesim(), unstablesim(), longstablesim(), solarsystemsim()], tmax=tmax)
        self.assertTrue(np.isnan(tinst[0]))
        self.assertLess(tinst[1], tmax[1])
        for i in range(2,5):
            self.assertEqual(tinst[i], tmax[i])

    def test_repeat(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        p1 = self.model.predict_stable(sim, tmax=1e4)
        p2 = self.model.predict_stable(sim, tmax=1e4)
        self.assertEqual(p1, p2)

    def test_hyperbolic(self):
        sim = hyperbolicsim()
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 0)
    
    def test_escape(self):
        sim = escapesim()
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 0)
    
    def test_unstable_in_short_integration(self):
        sim = unstablesim()
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 0)
    
    def test_solarsystem(self):
        sim = solarsystemsim()
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 1)
    
    def test_stable(self):
        sim = longstablesim()
        self.assertEqual(self.model.predict_stable(sim, tmax=1e4), 1)

    def test_same_traj_as_feature_classifier(self):
        # procedure in featureclassifier
        sim = unstablesim()
        init_sim_parameters(sim)                            
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)]
        featureargs = [10000, 80, trios]                    
        triofeatures, stable = features(sim, featureargs)   
        tfc = sim.t

        sim = unstablesim()
        init_sim_parameters(sim)
        try:
            sim.integrate(1e4*sim.particles[1].P, exact_finish_time=0)
        except rebound.Collision:
            tn = sim.t
        self.assertEqual(tfc, tn)

if __name__ == '__main__':
    unittest.main()
