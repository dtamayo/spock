# -*- coding: utf-8 -*-
"""Stability of Planetary Orbital Configurations Classifier"""
# Make changes for python 2 and 3 compatibility
try:
    import builtins      # if this succeeds it's python 3.x
    builtins.xrange = range
    builtins.basestring = (str,bytes)
except ImportError:
    pass                 # python 2.x

from .classifier import StabilityClassifier
from .nbody import Nbody

__all__ = ["StabilityClassifier", "Nbody"]
