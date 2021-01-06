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
import einops as E
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pytorch_lightning as pl
from .simsetup import init_sim_parameters
warnings.filterwarnings('ignore', "DeprecationWarning: Using or importing the ABCs")

profile = lambda _: _

def fast_truncnorm(
        loc, scale, left=np.inf, right=np.inf,
        d=10000, nsamp=50):
    """Fast truncnorm sampling.
    
    Assumes scale and loc have the desired shape of output.
    length is number of elements.
    Select nsamp based on expecting at minimum one sample of a Gaussian
        to fit within your (left, right) range.
    Select d based on memory considerations - need to operate on
        a (d, nsamp) array.
    """
    oldscale = scale
    oldloc = loc
    
    scale = scale.reshape(-1)
    loc = loc.reshape(-1)
    samples = np.zeros_like(scale)
    start = 0
        
    for start in range(0, scale.shape[0], d):

        end = start + d
        if end > scale.shape[0]:
            end = scale.shape[0]
        
        cd = end-start
        rand_out = np.random.normal(size=(nsamp, cd))

        rand_out = (
            rand_out * scale[None, start:end]
            + loc[None, start:end]
        )
        
        #rand_out is (nsamp, cd)
        if right == np.inf:
            mask = (rand_out > left)
        elif left == np.inf:
            mask = (rand_out < right)
        else:
            mask = (rand_out > left) & (rand_out < right)
            
        first_good_val = rand_out[
            mask.argmax(0), np.arange(cd)
        ]
        samples[start:end] = first_good_val
        
    return samples.reshape(*oldscale.shape)

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

    def predict(self, sim, samples=1000, indices=None, seed=0):
        """Estimate instability time for a given simulation.

        Computes the median of the posterior using the number of samples.

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test
        samples (int): Number of samples to return

        Returns:

        float: instability time in units of initial orbit
            of the innermost planet
        """

        samples = self.sample(sim, samples=samples, indices=indices, seed=seed)
        return np.median(samples)

    def resample_stable_sims(self, samps_time):
        """Use a prior fit to the unstable value histogram"""
        stable_past_9 = samps_time >= 9
        _prior = lambda logT: (
            3.27086190404742*np.exp(-0.424033970670719 * logT) -
            10.8793430454878*np.exp(-0.200351029031774 * logT**2)
        )
        normalization = quad(_prior, a=9, b=np.inf)[0]
        prior = lambda logT: _prior(logT)/normalization
        n_samples = stable_past_9.sum()
        bins = n_samples*4
        top = 100.
        bin_edges = np.linspace(9, top, num=bins)
        cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
        bin_edges = [9.] +list(bin_edges)+[top]
        inv_cdf = interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        samples = inv_cdf(r)
        samps_time[stable_past_9] = samples
        return samps_time

    @profile
    def sample(self, sim, samples=1000, indices=None, seed=0):
        """Return samples from a posterior over log instability time (base 10).
        This returns samples from a simple prior for all times greater than 10^9 orbits

        Parameters:

        sim (rebound.Simulation): Orbital configuration to test
        samples (int): Number of samples to return

        Returns:

        np.array: samples of the posterior
        """
        init_sim_parameters(sim)
        pl.seed_everything(seed)
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

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


        Xs = []
        for i, trio in enumerate(trios):
            sim = simt.copy()
            # These are the .npy.
            cur_tseries = tseries[None, i, ::10].astype(np.float32)
            mass_array = np.array([sim.particles[j].m/sim.particles[0].m for j in trio]).astype(np.float32)
            mass_array = E.repeat(mass_array, 'i -> () t i', t=100)
            X = data_setup_kernel(mass_array, cur_tseries)
            Xs.append(X)

        Xs = np.array(Xs)
        ntrios = Xs.shape[0]
        nt = 100
        X = E.rearrange(Xs, 'trio () time feature -> (trio time) feature')
        Xp = self.ssX.transform(X)
        Xp = E.rearrange(Xp, '(trio time) feature -> trio time feature',
                         trio=ntrios, time=nt)

        Xflat = torch.tensor(Xp).float()
        if self.cuda:
            Xflat = Xflat.cuda()

        sampled_mu_std = np.array([self.sample_full_swag(Xflat).detach().cpu().numpy() for _ in range(samples)])
        #(samples, trio, mu_std)

        samps_time = np.array(fast_truncnorm(
                sampled_mu_std[..., 0], sampled_mu_std[..., 1],
                left=4, d=10000, nsamp=40
            ))
        samps_time = self.resample_stable_sims(samps_time)
        outs = np.min(samps_time, 1)
        return 10.0**outs

@profile
def data_setup_kernel(mass_array, cur_tseries):
    """Data preprocessing"""
    nf = 32
    nt = 100
    concat_with_mass = np.concatenate((cur_tseries, mass_array), axis=2)
    assert concat_with_mass.shape == (1, nt, nf - 3)

    concat_with_nan = np.concatenate(
            (concat_with_mass,
            (~np.isfinite(concat_with_mass[..., [3, 6, 7]]))),
        axis=2)

    clean_input = np.nan_to_num(concat_with_nan, posinf=0.0, neginf=0.0)

    X = np.zeros((1, nt, nf + 9))

    cur_feature = 0
    for j in range(nf):
        if j in [11, 12, 13, 17, 18, 19, 23, 24, 25]: #if 'Omega' in label or 'pomega' in label or 'theta' in label:
            X[:, :, cur_feature] = np.cos(clean_input[:, :, j])
            cur_feature += 1
            X[:, :, cur_feature] = np.sin(clean_input[:, :, j])
            cur_feature += 1
        else:
            X[:, :, cur_feature] = clean_input[:, :, j]
            cur_feature += 1

    return X
