import rebound
from multiprocessing import Pool
from itertools import repeat
from runfunctions import run_random
import pandas as pd
import sys
import math

N_systems=2000
tmax = 1.e6
betamin=5
betamax=20

integrator=sys.argv[1]
dt = float(sys.argv[2])*math.sqrt(3)/100.
shadow = int(sys.argv[3])
Nthreads = int(sys.argv[4])
print(dt)

params = list(zip(range(N_systems), repeat(integrator), repeat(dt), repeat(tmax), repeat(betamin), repeat(betamax), repeat(shadow)))

pool = Pool(Nthreads)
res = pool.starmap(run_random, params)
df = pd.DataFrame(res, columns=('instability_time', 'frac_energy_error', 'wall_time'))

datapath = '../data/whfastias15comparison/'
if integrator=="ias15":
    filename = "ias15"
else:
    filename = "whfastdt{:.3f}".format(dt)
if shadow == 1:
    filename += "shadow"
filename += ".csv"

df.to_csv(datapath+filename, encoding='ascii')
