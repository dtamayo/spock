import dill
import pandas as pd
from  . import featurefunctions

class spockClassifier():
    def __init__(self, modelname='ressummaryfeaturesxgbv6_resonant1e+04.pkl'):
        self.model, self.features, featurefuncname = dill.load(open('../spock/models/'+modelname, "rb"))
        self.featurefunc = getattr(featurefunctions, featurefuncname)
    def predict(self, sim):
        summaryfeatures = self.featurefunc(sim)
        if self.features is not None:
            summaryfeatures = summaryfeatures[self.features] # take subset of features used in training the model
        summaryfeatures = pd.DataFrame([summaryfeatures]) # Put it in format model expects...would be nice to optimize out
        probstability = self.model.predict_proba(summaryfeatures)[:,1][0]
        return probstability
