
# Getting Started

## Installation

SPOCK is compatible with both Linux and Mac. 

Install with:

```
pip install spock
```

SPOCK relies on XGBoost, which has installation issues with OpenMP on Mac OSX. If you have problems (https://github.com/dmlc/xgboost/issues/4477), the easiest way is probably to install homebrew, and then:

```
brew install libomp
pip install spock
```


## Quickstart

Let's predict the probability that a given 3-planet system is stable
past 1 billion orbits with the XGBoost-based classifier, and then compute its
median expected instability time with the deep regressor:

```python
import rebound
from spock import FeatureClassifier, DeepRegressor
feature_model = FeatureClassifier()
deep_model = DeepRegressor()

sim = rebound.Simulation()
sim.add(m=1.)
sim.add(m=1.e-5, P=1., e=0.03, l=0.3)
sim.add(m=1.e-5, P=1.2, e=0.03, l=2.8)
sim.add(m=1.e-5, P=1.5, e=0.03, l=-0.5)
sim.move_to_com()

# XGBoost-based classifier
print(feature_model.predict_stable(sim))
# >>> 0.011505529

# Bayesian neural net-based regressor
median, lower, upper = deep_model.predict_instability_time(sim, samples=10000)
print(int(median))
# >>> 419759
```

