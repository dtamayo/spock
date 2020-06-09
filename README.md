# SPOCK 🖖 
*Stability of Planetary Orbital Configurations Klassifier*

## Quickstart

Let's predict the probaility that a given 3-planet system is stable:

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

model.predict_stable(sim)
>>> 0.011536411
```

## Examples

The best place to start is the example notebooks in jupyter\_examples/

## Installation

```shell
pip install spock
```

SPOCK relies on XGBoost, which has installation issues with OpenMP on Mac OSX. If you have problems (<https://github.com/dmlc/xgboost/issues/4477>), the easiest way is probably to install [homebrew](brew.sh), and

```shell
brew install libomp
pip install spock
```
