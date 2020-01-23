from subprocess import call
from collections import OrderedDict
import sys

datapath = '../training_data/'

datasets = 'all' # either a list of folders ([resonant, TTVsystems/Kepler-431]) or 'all' or 'ttv' to expand
runfunc = 'features'#'orbtseries'#'orbsummaryfeaturesxgb'

kwargs = OrderedDict()
kwargs['Norbits'] = 1e4
kwargs['Nout'] = 80
kwargs['trio'] = [[1,2,3]]

foldername = runfunc
for key, val in kwargs.items():
    if key == 'trio':
        foldername += 'trio'
    else:
        foldername += '{0}{1}'.format(key, val)

if datasets == 'all':
    datasets = ['random', 'resonant', 'TTVsystems/KOI-1576/', 'nonressystems/Kepler-431/']

for dataset in list(datasets):
    folder = datapath + dataset + '/'
    #call('rm -f ' + folder + 'labels.csv', shell=True)
    #call('rm -f ' + folder + 'massratios.csv', shell=True)
    #call('rm -f ' + folder + 'runstrings.csv', shell=True)
    print('removing ' + folder + foldername)
    call('rm -rf "' + folder + foldername + '"', shell=True)
    call('rm -rf "' + foldername + '"', shell=True)
