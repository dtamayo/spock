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
from .spock_reg_model import load_swag
from pytorch_lightning import Trainer
import torch
import time
from numba import jit
import pickle as pkl
import warnings
warnings.filterwarnings('ignore', "DeprecationWarning: Using or importing the ABCs")

profile = lambda _: _

class FeatureRegressor(object):
    def __init__(self, cuda=False, filebase='*v50_*output.pkl'):
        super(FeatureRegressor, self).__init__()
        pwd = os.path.dirname(__file__)
        pwd = pwd + '/models/regression'
        self.cuda = cuda

        #Load model
        self.swag_ensemble = [
            load_swag(fname).cpu()
            for i, fname in enumerate(glob.glob(pwd + '/' + filebase)) #0.78, 0.970
        ]
        self.ssX = StandardScaler()

        #Data scaling parameters:
        self.ssX.scale_ = np.array([2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
                   6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
                   4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
                   7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
                   7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
                   7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
                   7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
                   2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
                   7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
                   2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
                   7.11351099e-02])
        self.ssX.mean_ = np.array([ 4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
                   6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
                   9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
                  -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
                   6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
                   7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
                  -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
                   2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
                  -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
                   1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
                   5.08607199e-03])
        self.ssX.var_ = self.ssX.scale_**2

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
