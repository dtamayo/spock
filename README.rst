SPOCK 🖖 
********
**Stability of Planetary Orbital Configurations Klassifier**

.. image:: https://badge.fury.io/py/spock.svg
    :target: https://badge.fury.io/py/spock
.. image:: https://travis-ci.com/dtamayo/spock.svg?branch=master
    :target: https://travis-ci.com/dtamayo/spock
.. image:: http://img.shields.io/badge/license-GPL-green.svg?style=flat 
    :target: https://github.com/dtamayo/spock/blob/master/LICENSE
.. image:: https://img.shields.io/badge/launch-binder-ff69b4.svg?style=flat
    :target: http://mybinder.org/repo/dtamayo/spock


Quickstart
==========

Let's predict the probaility that a given 3-planet system is stable:

.. code:: python

    python
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

Examples
========

The best place to start is the example notebooks in `jupyter\_examples/ <https://github.com/dtamayo/spock/tree/master/jupyter_examples>`_.

Installation
============

Install with::

    pip install spock

SPOCK relies on XGBoost, which has installation issues with OpenMP on Mac OSX. If you have problems (`https://github.com/dmlc/xgboost/issues/4477 <https://github.com/dmlc/xgboost/issues/4477>`_), the easiest way is probably to install `homebrew <brew.sh>`_, and::

    brew install libomp
    pip install spock
