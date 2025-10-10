# SPOCK 🖖

**Stability of Planetary Orbital Configurations Klassifier**

[![image](https://badge.fury.io/py/spock.svg)](https://badge.fury.io/py/spock)
[![image](http://img.shields.io/badge/license-GPL-green.svg?style=flat)](https://github.com/dtamayo/spock/blob/master/LICENSE)
[![image](http://img.shields.io/badge/arXiv-2007.06521-green.svg?style=flat)](http://arxiv.org/abs/2007.06521)
[![image](http://img.shields.io/badge/arXiv-2101.04117-green.svg?style=flat)](https://arxiv.org/abs/2101.04117)
[![image](http://img.shields.io/badge/arXiv-2106.14863-green.svg?style=flat)](https://arxiv.org/abs/2106.14863)
![image](https://raw.githubusercontent.com/dtamayo/spock/master/paper_plots/spockpr.jpg)

[Documentation](https://spock-instability.readthedocs.io/en/latest/)

The SPOCK package incorporates several machine learning and analytical tools for estimating the stability of compact planetary configurations.
All estimators use a common API to facilitate comparisons between them and with N-body integrations.
SPOCK has been updated to handle the edge cases of systems with 0-2 planets (2-planet systems evaluated using Hill stability). See [changelog.md](https://github.com/dtamayo/spock/blob/master/changelog.md).

# Quickstart

Let's predict the probability that a given 3-planet system is stable past 1 billion orbits with the XGBoost-based classifier of [Tamayo et al., 2020](http://arxiv.org/abs/2007.06521).

```python
import rebound
from spock import FeatureClassifier
feature_model = FeatureClassifier()

sim = rebound.Simulation()
sim.add(m=1.)
sim.add(m=1.e-5, P=1., e=0.03, pomega=2., l=0.5)
sim.add(m=1.e-5, P=1.2, e=0.03, pomega=3., l=3.)
sim.add(m=1.e-5, P=1.5, e=0.03, pomega=1.5, l=2.)
sim.move_to_com()

print(feature_model.predict_stable(sim))
# >>> 0.06591137
```

This model provides a simple scalar probability of stability over a billion orbits.
We can instead estimate its median expected instability time using the deep regressor from [Cranmer et al., 2021](https://arxiv.org/abs/2101.04117).

```python
import numpy as np
from spock import DeepRegressor
deep_model = DeepRegressor()

median, lower, upper, samples = deep_model.predict_instability_time(
    sim, samples=10000, return_samples=True, seed=0
)
print(10**np.average(np.log10(samples)))  # Expectation of log-normal
# >>> 414208.4307974086

print(median)
# >>> 223792.38826507595
```

The returned time is expressed in the time units used in setting up the REBOUND Simulation above.
Since we set the innermost planet orbit to unity, this corresponds to 242570 innermost planet orbits.

We can compare these results to the semi-analytic criterion of [Tamayo et al., 2021](https://arxiv.org/abs/2106.14863) for how likely the configuration is to be dynamically chaotic.
This is not a one-to-one comparison, but configurations that are chaotic through two-body MMR overlap are generally unstable on long timescales (see paper and examples).

```python
from spock import AnalyticalClassifier
analytical_model = AnalyticalClassifier()

print(analytical_model.predict_stable(sim))
# >>> 0.0
```

To match up with the above classifiers, the analytical classifier returns the probability the configuration is *regular*, i.e., not chaotic.
A probability of zero therefore corresponds to confidently chaotic.

We can also predict the collisional outcome of this system using the MLP model from [Lammers et al., 2024](https://arxiv.org/abs/2408.08873).

```python
from spock import CollisionMergerClassifier
class_model = CollisionMergerClassifier()

prob_12, prob_23, prob_13 = class_model.predict_collision_probs(sim)

print(prob_12, prob_23, prob_13)
# >>> 0.2738345 0.49277353 0.23339202
```

This model returns the probability of a physical collision occurring between planets 1 & 2, 2 & 3, and 1 & 3 when provided a three-planet system. In this case, the instability will most likely result in a collision between planets 2 & 3, but all three outcomes are possible (ejections are exceedingly rare in compact multiplanet systems).

Additionally, we can predict the states of the post-collision planets using the model from [Lammers et al., 2024](https://arxiv.org/abs/2408.08873).

```python
from spock import CollisionOrbitalOutcomeRegressor
reg_model = CollisionOrbitalOutcomeRegressor()

new_sim = reg_model.predict_collision_outcome(sim, collision_inds=[2, 3])

print(new_sim)
# >>> <rebound.simulation.Simulation object at 0x303ed70d0, N=3, t=0.0>
```

Note that this model makes the usual assumption that mergers are perfectly inelastic.

These models are conveniently combined into a giant impact emulator, which predicts instability times, predicts and samples collision pair probabilities, and then handles the collisions. This is repeated until SPOCK predicts the system to be stable on a user-specified timescale (the default is a billion orbits).

```python
from spock import GiantImpactPhaseEmulator
emulator = GiantImpactPhaseEmulator()

new_sim = emulator.predict(sim)

print(new_sim)
# >>> <rebound.simulation.Simulation object at 0x303f05c50, N=3, t=999999999.9999993>
```

Only one collision takes place in this example system, but the typical use case involves starting with more bodies (~10).
See [this example](https://github.com/dtamayo/spock/blob/master/jupyter_examples/QuickStart.ipynb) for additional information about the models included in SPOCK, and see [jupyter\_examples/](https://github.com/dtamayo/spock/tree/master/jupyter_examples) for more thorough example applications.

# Examples

[Colab tutorial](https://colab.research.google.com/drive/1R3NrPmtI5DZFq_VZtv8gowINBrXM85Zv?usp=sharing)
for the deep regressor.

The example notebooks contain many additional examples:
[jupyter\_examples/](https://github.com/dtamayo/spock/tree/master/jupyter_examples).

# Installation

SPOCK is compatible with both Linux and Mac. SPOCK relies on XGBoost, which has installation issues with OpenMP on
Mac OSX. If you have problems (<https://github.com/dmlc/xgboost/issues/4477>), the easiest way is
probably to install [homebrew](brew.sh), and:

```
brew install libomp
```

The most straightforward way to avoid any version conflicts is to download the Anaconda Python distribution and make a separate conda environment.

Here we create we create a new conda environment called `spock` and install all the required dependencies
```
conda create -q --name spock -c pytorch -c conda-forge python=3.13 numpy scipy pandas scikit-learn matplotlib torchvision pytorch xgboost rebound einops jupyter pytorch-lightning ipython h5py
conda activate spock
pip install spock
```

Each time you want to use spock you will first have to activate this `spock` conda environment (google conda environments).
