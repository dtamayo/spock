import rebound
import numpy as np
from subprocess import call
import sys
import warnings
warnings.filterwarnings('ignore') # to suppress warnings about REBOUND versions that I've already tested
from collections import OrderedDict
from training_data_functions import gen_training_data
sys.path.append('../spock')
from feature_functions import features
from additional_feature_functions import additional_features

runfunc = features
datapath = '../data/'
repopath = '../'

datasets = 'all' # either a list of folders ([resonant, TTVsystems/Kepler-431]) or 'all' or 'ttv' to expand

kwargs = OrderedDict()
kwargs['Norbits'] = 1e4
kwargs['Nout'] = 80 
kwargs['trio'] = [[1,2,3]]

foldername = runfunc.__name__
for key, val in kwargs.items():
    if key == 'trio':
        foldername += 'trio'
    else:
        foldername += '{0}{1}'.format(key, val)

gendatafolder = repopath + 'generate_training_data/'

already_exists = call('mkdir "' + gendatafolder + foldername + '"', shell=True)
if not already_exists: # store a copy of this script in generate_data so we can always regenerate what we had
    call('cp "' + gendatafolder + '/generate_data.py" "' + gendatafolder + foldername + '"', shell=True)
    call('python "' + gendatafolder + foldername + '/generate_data.py"' , shell=True)
    exit()
    # we always run the copied script so that we do the same thing whether we're running for first time or regenerating
# if it does exist don't overwrite since we don't want to overwrite history

if datasets == 'all':
    datasets = ['random', 'resonant', 'TTVsystems/KOI-1576/', 'nonressystems/Kepler-431/']

for dataset in list(datasets):
    if dataset == 'random':
        if rebound.__githash__ != '4a6c79ae14ffde27828dd9d1f8d8afeba94ef048':
            print('random dataset not run. Check out rebound commit 4a6c79ae14ffde27828dd9d1f8d8afeba94ef048 (HEAD of spockrandomintegrations branch on dtamayo/rebound fork) and rerun script if needed')
            continue 
    else:
        if rebound.__githash__ != '6fb912f615ca542b670ab591375191d1ed914672':
            print('{0} dataset not run. Check out rebound commit 6fb912f615ca542b670ab591375191d1ed914672 and rerun script if needed'.format(dataset))
            continue 

    safolder = datapath + dataset + '/simulation_archives/runs/'
    trainingdatafolder = repopath + 'training_data/' + dataset + '/'

    already_exists = call('mkdir "' + trainingdatafolder + foldername + '"', shell=True)
    if already_exists:
        print('output folder already exists at {0}. Remove it if you want to regenerate'.format(trainingdatafolder+foldername))
        continue
    call('cp "' + trainingdatafolder + 'labels.csv" "' + trainingdatafolder + foldername + '"', shell=True)
    call('cp "' + trainingdatafolder + 'massratios.csv" "' + trainingdatafolder + foldername + '"', shell=True)
    call('cp "' + trainingdatafolder + 'runstrings.csv" "' + trainingdatafolder + foldername + '"', shell=True)

    print(trainingdatafolder + foldername)
    gen_training_data(trainingdatafolder + foldername, safolder, runfunc, list(kwargs.values()))
