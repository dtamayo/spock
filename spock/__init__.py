# -*- coding: utf-8 -*-
"""Stability of Planetary Orbital Configurations Klassifier"""

from .featureclassifier import FeatureClassifier
from .nbodyregressor import NbodyRegressor
from .version import __version__
try:
    from .version import __githash__
except:
    pass

__all__ = ["FeatureClassifier", "NbodyRegressor"]
