import rebound
import pandas as pd
if rebound.__githash__ != '6fb912f615ca542b670ab591375191d1ed914672':
    print('Check out rebound commit 6fb912f615ca542b670ab591375191d1ed914672 and rerun script. See spock/training.md')
    [
datapath = '../data/'

randompath = 'random/simulation_archives/runs/'
respath = 'resonant/simulation_archives/runs/'

dfrand = pd.read_csv('../training_data/random/labels.csv', index_col=0)
dfres = pd.read_csv('../training_data/resonant/labels.csv', index_col=0)

def randPratios(row):
    sa = rebound.SimulationArchive(datapath+randompath+'sa{0}'.format(row['runstring']))
    sim = sa[0]
    ps = sim.particles
    row['Pratio21'] = ps[2].P/ps[1].P
    row['Pratio32'] = ps[3].P/ps[2].P
    RH12 = ps[1].a*((ps[1].m+ps[2].m)/3.)**(1/3)
    RH23 = ps[2].a*((ps[2].m+ps[3].m)/3.)**(1/3)
    row['beta12'] = (ps[2].a-ps[1].a)/RH12
    row['beta23'] = (ps[3].a-ps[2].a)/RH23
    return row

def resPratios(row):
    sa = rebound.SimulationArchive(datapath+respath+'sa{0}'.format(row['runstring']))
    sim = sa[0]
    ps = sim.particles
    row['Pratio21'] = ps[2].P/ps[1].P
    row['Pratio32'] = ps[3].P/ps[2].P
    RH12 = ps[1].a*((ps[1].m+ps[2].m)/3.)**(1/3)
    RH23 = ps[2].a*((ps[2].m+ps[3].m)/3.)**(1/3)
    row['beta12'] = (ps[2].a-ps[1].a)/RH12
    row['beta23'] = (ps[3].a-ps[2].a)/RH23
    return row

dfrand = dfrand.apply(randPratios, axis=1)
dfrand.to_csv('randomPratios.csv')

dfres = dfres.apply(resPratios, axis=1)
dfres.to_csv('resPratios.csv')
