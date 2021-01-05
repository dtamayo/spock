import numpy as np
from scipy.stats import truncnorm
import os
from collections import OrderedDict
from .feature_functions import features
from .tseries_feature_functions import get_extended_tseries
from copy import deepcopy as copy
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.functional import F
from icecream import ic
import glob
import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
import torch
import time
from numba import jit
import pickle as pkl

profile = lambda _: _

models/regression/
class FeatureRegressor(object):
    def __init__(self, cuda=False, filebase='*v50_*.pkl')
        super(FeatureRegressor, self).__init__()
        pwd = os.path.dirname(__file__)
        pwd = pwd + '/models/regression'
        self.cuda = cuda

        #Load model
        self.swag_ensemble = [
            spock_reg_model.load_swag(fname).cpu()
            for i, fname in enumerate(glob.glob(pwd + '/' + filebase)) #0.78, 0.970
        ]
        ssX_file = basedir_bayes + '/' + filebase[:-4] + '_ssX.pkl'
        ic(ssX_file)
        self.ssX = pkl.load(open(list(glob.glob(ssX_file))[0], 'rb'))
        #Load data scaling parameters

    def sample_full_swag(self, X_sample):
        """Pick a random model from the ensemble and sample from it
            within each model, it samples from its weights."""
        
        swag_i = np.random.randint(0, len(self.swag_ensemble))
        swag_model = self.swag_ensemble[swag_i]
        swag_model.eval()
        if self.cuda:
            swag_model.w_avg = swag_model.w_avg.cuda()
            swag_model.w2_avg = swag_model.w2_avg.cuda()
            swag_model.pre_D = swag_model.pre_D.cuda()
            swag_model.cuda()
        out = swag_model.forward_swag_fast(X_sample, scale=0.5)
        if self.cuda:
            swag_model.w_avg = swag_model.w_avg.cpu()
            swag_model.w2_avg = swag_model.w2_avg.cpu()
            swag_model.pre_D = swag_model.pre_D.cpu()
            swag_model.cpu()
        return out

    def predict(self, sim, indices=None, samples=1000):
        """Estimate instability time for a given simulation.

        :sim: The rebound simulation.
        :indices: The list of planets to consider.
        :samples: How many MC samples to return.
        :returns: Array of samples of log10(T) for the simulation.
            The spread of samples covers both epistemic
            (model-based) and aleatoric (real, data-based) uncertainty.
            Samples above log10(T) indicate a stable simulation. Bounded
            between 4 and 12.

        """
        samples = self.sample(sim, indices, samples)
        return np.median(samples)

    @profile
    def sample(self, sim, indices=None, samples=1000):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

        ic(trios)
        kwargs = OrderedDict()
        kwargs['Norbits'] = int(1e4)
        kwargs['Nout'] = 1000
        kwargs['trios'] = trios
        args = list(kwargs.values())
        # These are the .npy.
        # In the other file, we concatenate (restseries, orbtseries, mass_array)
        tseries, stable = get_extended_tseries(sim, args)

        if not stable:
            return None

        tseries = np.array(tseries)
        simt = sim.copy()
        alltime = []
        for i, trio in enumerate(trios):
            sim = simt.copy()
            # These are the .npy.
            # In the other file, we concatenate (restseries, orbtseries, mass_array)
            cur_tseries = tseries[None, i, ::10]
            mass_array = np.array([sim.particles[j].m/sim.particles[0].m for j in trio])
            X = data_setup_kernel(mass_array, cur_tseries)
            X = self.ssX.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X = torch.tensor(X).float()
            if self.cuda:
                X = X.cuda()

            time = torch.cat([self.sample_full_swag(X)[None] for _ in range(samples)], dim=0).detach()

            if self.cuda:
                time = time.cpu()

            time = time.numpy()
            alltime.append(time)

        out = np.array(alltime)[..., 0, :]
        mu = out[..., 0]
        std = out[..., 1]

        #Old:
        # a = (4 - mu)/std
        # b = np.inf
        # try:
            # samples = truncnorm.rvs(a, b)*std + mu
        # except ValueError:
            # return None
        # first_inst = np.min(samples, 0)
        # return first_inst

        #HACK - should this be inf at the top?
        # a, b = (4 - out[..., 0]) / out[..., 1], np.inf #(12 - out[..., 0]) / out[..., 1]
        # try:
            # samples = truncnorm.rvs(a, b, loc=out[..., 0], scale=out[..., 1])
        # except ValueError:
            # return None
        # return np.min(samples, 0)

        return mu, std

@jit
def data_setup_kernel(mass_array, cur_tseries):
    mass_array = np.tile(mass_array[None], (100, 1))[None]

    old_X = np.concatenate((cur_tseries, mass_array), axis=2)

    isnotfinite = lambda _x: ~np.isfinite(_x)

    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [3]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [6]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [7]]).astype(np.float)), axis=2)

    old_X[..., :] = np.nan_to_num(old_X[..., :], posinf=0.0, neginf=0.0)

    # axis_labels = []
    X = []

    for j in range(old_X.shape[-1]):#: #, label in enumerate(old_axis_labels):
        if j in [11, 12, 13, 17, 18, 19, 23, 24, 25]: #if 'Omega' in label or 'pomega' in label or 'theta' in label:
            X.append(np.cos(old_X[:, :, [j]]))
            X.append(np.sin(old_X[:, :, [j]]))
            # axis_labels.append('cos_'+label)
            # axis_labels.append('sin_'+label)
        else:
            X.append(old_X[:, :, [j]])
            # axis_labels.append(label)
    X = np.concatenate(X, axis=2)
    if X.shape[-1] != 41:
        raise NotImplementedError("Need to change indexes above for angles, replace ssX.")

    return X
