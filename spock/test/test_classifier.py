import rebound
import unittest
from spock import StabilityClassifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.model = StabilityClassifier()

    def test_negmass(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, P=1.)
        sim.add(m=-1.e-5, P=2.)
        sim.add(m=1.e-5, P=3.)
        with self.assertRaises(AttributeError):
            self.model.predict_stable(sim)
        
    def test_hyperbolic(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-5, a=-1., e=1.2)
        sim.add(m=1.e-5, a=2.)
        sim.add(m=1.e-5, a=3.)
        self.assertEqual(self.model.predict_stable(sim), 0)
    
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
    
if __name__ == '__main__':
    unittest.main()
