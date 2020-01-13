from subprocess import call
from collections import OrderedDict
import sys

datapath = '../data/'

datasets = 'all' # either a list of folders ([resonant, TTVsystems/Kepler-431]) or 'all' or 'ttv' to expand
runfunc = 'spock_features'#'orbtseries'#'orbsummaryfeaturesxgb'

kwargs = OrderedDict()
kwargs['Norbits'] = 1e4
kwargs['Nout'] = 1000
kwargs['trio'] = [[1,2,3]]

foldername = runfunc
for key, val in kwargs.items():
    if key == 'trio':
        foldername += 'trio'
    else:
        foldername += '{0}{1}'.format(key, val)

def allsystems():
    return ['random', 'resonant'] + ttvsystems() + nonressystems()

def ttvsystems():
    folders = ['KOI-1576']
    return ['TTVsystems/' + folder for folder in folders]

def nonressystems():
    folders = ['Kepler-431']
    return ['nonressystems/' + folder for folder in folders]

if datasets == 'all':
    datasets = allsystems()

if datasets == 'ttv':
    datasets = ttvsystems()

if datasets == 'nonres':
    datasets = nonressystems()

for dataset in list(datasets):
    folder = datapath + dataset + '/'
    #call('rm -f ' + folder + 'labels.csv', shell=True)
    #call('rm -f ' + folder + 'massratios.csv', shell=True)
    #call('rm -f ' + folder + 'runstrings.csv', shell=True)
    print('removing ' + folder + foldername)
    call('rm -rf "' + folder + foldername + '"', shell=True)
    call('rm -rf "' + foldername + '"', shell=True)
