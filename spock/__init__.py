# -*- coding: utf-8 -*-
"""Stability of Planetary Orbital Configurations Klassifier"""

from .analyticalclassifier import AnalyticalClassifier
from .deepregressor import DeepRegressor
from .featureclassifier import FeatureClassifier
from .nbodyregressor import NbodyRegressor
from .collision_merger_classifier import class_MLP, CollisionMergerClassifier
from .collision_orbital_outcome_regressor import reg_MLP, CollisionOrbitalOutcomeRegressor
from .giant_impact_phase_emulator import GiantImpactPhaseEmulator
try:
    from .version import __version__ # in case _version.py not available 
except:
    __version__ = "unknown version"

__all__ = ["FeatureClassifier", "DeepRegressor", "NbodyRegressor", "AnalyticalClassifier", "class_MLP", "reg_MLP", "CollisionMergerClassifier", "CollisionOrbitalOutcomeRegressor", "GiantImpactPhaseEmulator"]
