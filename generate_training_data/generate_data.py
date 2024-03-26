import rebound
import numpy as np
from subprocess import call
import os
import sys
import warnings
warnings.filterwarnings('ignore') # to suppress warnings about REBOUND versions that I've already tested
from collections import OrderedDict
sys.path.append(os.path.join(sys.path[0], '..'))
from training_data_functions import gen_training_data
from feature_functions import features
from additional_feature_functions import additional_features

runfunc = features
csvpath = '../csvs/'
repopath = '../'

datasets = ['resonant']#'all' # either a list of folders ([resonant, TTVsystems/Kepler-431]) or 'all' or 'ttv' to expand

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
    datasets = ['random', 'resonant']#, 'TTVsystems/KOI-1576/', 'nonressystems/Kepler-431/']

for dataset in list(datasets):
    outputfolder = repopath + 'training_data/' + dataset + '/' + foldername
    csvfolder = csvpath + '/' + dataset + '/'

    already_exists = call('mkdir "' + outputfolder + '"', shell=True)
    if already_exists:
        print('output folder already exists at {0}. Remove it if you want to regenerate'.format(outputfolder))
        continue
    
    gen_training_data(outputfolder, csvfolder, runfunc, list(kwargs.values()))
