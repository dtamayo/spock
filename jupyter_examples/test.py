from spock import DeepRegressor2
import rebound
import numpy as np
import torch

def init_process():
    global model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = DeepRegressor2()
    
def predtime(params):
    Pratio, e = params
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1.e-5, P=1, e=0)
    sim.add(m=1.e-5, P=Pratio, e=e)
    sim.add(m=1.e-5, P=Pratio**2, e=0)
    tinst, _, _ = model.predict_instability_time(sim, samples=100)
    return tinst

def get_pool_params(xlist, ylist):
    params = []
    for y in ylist:
        for x in xlist:
            params.append((x, y))
    return params

if __name__ == "__main__":
    es = np.linspace(0, 0.1, 8)
    Pratios = np.linspace(1.15, 1.3, 10)
    params = get_pool_params(Pratios, es)
    torch.multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(processes=64, initializer=init_process) as pool:
        tinsts = pool.map(predtime, params)
    np.save('tinsts.npy', tinsts)