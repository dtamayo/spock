from spock import featureclassifier

#this file is to get around the jupyter notebook bug with multiprocessing

spock = featureclassifier.FeatureClassifier()

def getSpock(sim):
    return spock.generate_features(sim)
