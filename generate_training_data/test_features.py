import rebound
from spock import FeatureClassifier

model = None
def init_model():
    global model
    model = FeatureClassifier()

def getFeat(num):
    '''when given a index of a row, loads initial conditions and returns the spock generated features'''
    #gets features based on index num
    sim = rebound.Simulation("../dataset/resonant/clean_initial_conditions.bin", snapshot=num)
    return model.generate_features(sim)

def test(i):
    return i
