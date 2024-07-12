import spock
from spock import featureKlassifier
from spock import featureclassifier

#this file is to get around the jupyter notebook bug with multiprocessing

new = featureKlassifier.FeatureKlassifier()
old = featureclassifier.FeatureClassifier()

def getNew(sim):
    return new.generate_features(sim)
def getOld(sim):
    return old.generate_features(sim)