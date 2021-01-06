import rebound
import unittest
from spock import NbodyRegressor, DeepRegressor
import numpy as np

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
        p1 = self.model.predict(sim, seed=0)
        p2 = self.model.predict(sim, seed=1)
        print(p1, p2)
        self.assertTrue(self.relative(p1, p2) < 0.1)
        self.assertTrue(self.relative(p1, p2) > 0.0)

   
    def test_prediction(self):
        times = []
        for mass in [1e-4, 5e-5, 3e-5, 1e-5]:
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=mass, P=1)
            sim.add(m=mass, P=1.3)
            sim.add(m=mass, P=1.6)
            predicted_time = self.model.predict(sim)
            times.append(predicted_time)

        times = np.array(times)
        print(times)

        # First one is unstable:
        self.assertTrue(times[0] < 4.0)
        # Should get more stable:
        self.assertTrue(np.all(times[1:] > times[:-1]))
    
    
if __name__ == '__main__':
    unittest.main()
