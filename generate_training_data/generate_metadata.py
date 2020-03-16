import rebound
import numpy as np
import pandas as pd
import os
from subprocess import call
import warnings
warnings.filterwarnings('ignore') # filter REBOUND warnings about version that I've already tested

datapath = '../data/'
repopath = '../'

if rebound.__githash__ != '6fb912f615ca542b670ab591375191d1ed914672':
    print('Need to install REBOUND commit 6fb912f615ca542b670ab591375191d1ed914672 to ensure this runs correctly')
    exit()

def labels(row):
    try:
        sa = rebound.SimulationArchive(pathtosa+'sa'+row['runstring'])
        sim = sa[0]
        P1 = sim.particles[1].P # Need initial orbital period for TTVsystems, where P1 != 1

        try: # Needed for old integrations (random and Naireen) because no snapshot on end
            sim = rebound.Simulation(pathtosa+'../../final_conditions/runs/fc'+row['runstring'])
            row['instability_time'] = sim.t/P1
            try:
                ssim = rebound.Simulation(pathtossa+'../../final_conditions/shadowruns/fc'+row['runstring'])
                row['shadow_instability_time'] = ssim.t/P1
            except:
                print('No shadow for random {0}'.format(row['runstring']))
        except: # New runs (resonant and Ari) have snapshots at collision
            sim = sa[-1]
            if sim.t > 9.99e3 and sim.t < 1.0001e4: # catch all stable integrations accidentally thrown out. Have checked this catches all thrown out, and doesn't catch any with actual  instability times in this range
                row['instability_time'] = 1e9
                row['shadow_instability_time'] = 1e9
            else:
                row['instability_time'] = sim.t/P1
                ssa = rebound.SimulationArchive(pathtossa+'sa'+row['runstring'])
                ssim = ssa[-1]
                row['shadow_instability_time'] = ssim.t/P1
        row['Stable'] = row['instability_time'] > 9.99e8
    except:
        print('Error', pathtosa+'sa'+row['runstring'])
    return row

def massratios(row):
    try:
        sa = rebound.SimulationArchive(pathtosa+'sa'+row['runstring'])
        sim = sa[0]
        row['m1'] = sim.particles[1].m/sim.particles[0].m
        row['m2'] = sim.particles[2].m/sim.particles[0].m
        row['m3'] = sim.particles[3].m/sim.particles[0].m
    except:
        print(pathtosa+'sa'+row['runstring'])
    return row

def ttvsystems():
    folders = ['KOI-1576']
    return ['TTVsystems/' + folder for folder in folders]

def nonressystems():
    folders = ['Kepler-431']
    return ['nonressystems/' + folder for folder in folders]


datasets = ['resonant', 'random'] + ttvsystems() + nonressystems()
for dataset in datasets:
    print(dataset)
    pathtosa = datapath + dataset + '/simulation_archives/runs/'
    pathtossa = datapath + dataset + '/simulation_archives/shadowruns/'
    pathtotraining = repopath + 'training_data/' + dataset + '/'

    root, dirs, files = next(os.walk(pathtosa))
    runstrings = [file[2:] for file in files if file[-3:] == 'bin']
    df = pd.DataFrame(runstrings, columns=['runstring'])
    df = df.sort_values(by='runstring')
    df = df.reset_index(drop=True)
    df.to_csv(pathtotraining+'runstrings.csv', encoding='ascii')

    df['instability_time'] = -1
    df['shadow_instability_time'] = -1
    df['Stable'] = -1

    df = df.apply(labels, axis=1)
    df.to_csv(pathtotraining+'labels.csv', encoding='ascii')
    df = pd.DataFrame(df['shadow_instability_time'])
    call('mkdir ' + pathtotraining + 'shadowtimes', shell=True)
    call('cp ' + pathtotraining + 'labels.csv ' + pathtotraining + 'shadowtimes/', shell=True)
    df.to_csv(pathtotraining+'shadowtimes/trainingdata.csv', encoding='ascii')

    df = pd.read_csv(pathtotraining+'runstrings.csv', index_col=0)
    df['m1'] = -1
    df['m2'] = -1
    df['m3'] = -1

    df = df.apply(massratios, axis=1)
    df.to_csv(pathtotraining+'massratios.csv', encoding='ascii')
    call('cp ' + pathtotraining + 'massratios.csv ' + pathtotraining + 'shadowtimes/', shell=True)
    call('cp ' + pathtotraining + 'runstrings.csv ' + pathtotraining + 'shadowtimes/', shell=True)
