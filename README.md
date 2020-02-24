# SPOCK 🖖 
*Stability of Planetary Orbital Configurations Klassifier*

## Quickstart

Let's predict the stability of a given 3-planet system:

```python
import rebound
from spock import StabilityClassifier
model = StabilityClassifier()
sim = rebound.Simulation()
sim.add(m=1.)
sim.add(m=1.e-5, P=1., e=0.03, l=0.3)
sim.add(m=1.e-5, P=1.2, e=0.03, l=2.8)
sim.add(m=1.e-5, P=1.5, e=0.03, l=-0.5)
sim.move_to_com()
model.predict(sim, copy=False)
>>> 0.0048521925
```

