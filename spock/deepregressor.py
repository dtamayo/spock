import numpy as np
import math
from scipy.stats import truncnorm
import os
from collections import OrderedDict
from .tseries_feature_functions import get_extended_tseries
from copy import deepcopy as copy
import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.functional import F
import glob
from .spock_reg_model import load_swag
from pytorch_lightning import Trainer
import torch
import time
import pickle as pkl
import warnings
import einops as E
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pytorch_lightning as pl
from .simsetup import init_sim_parameters
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
import rebound
import random
warnings.filterwarnings('ignore', "DeprecationWarning: Using or importing the ABCs")

def fitted_prior():
    """A prior that was fit to the PDF of instability times in the training set below T=9"""
    return (lambda logT: 3.27086190404742*np.exp(-0.424033970670719 * logT) - 10.8793430454878*np.exp(-0.200351029031774 * logT**2))

def flat_prior(upper_limit):
    """A uniform prior between 9 and an upper limit"""
    return (lambda logT: 1.0 * (logT <= upper_limit))

def exponential_decaying_prior(decay_rate):
    """An exponentially decaying prior, which decays at a rate e^(-T*decay_rate)"""
    return (lambda logT: np.exp(-decay_rate * (logT - 9)))

profile = lambda _: _

def generate_dataset(sim): 
    sim = sim.copy()
    init_sim_parameters(sim, megno=False, safe_mode=0)
    if sim.N_real < 4:
        raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
    trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

    kwargs = OrderedDict()
    kwargs['Norbits'] = int(1e4)
    kwargs['Nout'] = 100
    kwargs['trios'] = trios
    args = list(kwargs.values())
    # These are the .npy.
    # In the other file, we concatenate (restseries, orbtseries, mass_array)
    tseries, stable = get_extended_tseries(sim, args, mmr=False, megno=False)

    if stable != True:
        time = stable
        return time

    tseries = np.array(tseries)
    alltime = []

    Xs = []
    for i, trio in enumerate(trios):
        # These are the .npy.
        cur_tseries = tseries[None, i, :].astype(np.float32)
        mass_array = np.array([sim.particles[j].m/sim.particles[0].m for j in trio]).astype(np.float32)
        mass_array = E.repeat(mass_array, 'i -> () t i', t=100)
        X = data_setup_kernel(mass_array, cur_tseries)
        Xs.append(X)

    Xs = np.array(Xs)
    return Xs

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
    t_inst_samples = np.zeros_like(scale)
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
        t_inst_samples[start:end] = first_good_val
        
    return t_inst_samples.reshape(*oldscale.shape)

class DeepRegressor(object):
    def __init__(self, cuda=False, filebase='*v50_*output.pkl'):
        super(DeepRegressor, self).__init__()
        pwd = os.path.dirname(__file__)
        pwd = pwd + '/models/regression'
        self.cuda = cuda

        #Load model
        self.swag_ensemble = [
            load_swag(fname).cpu()
            for i, fname in enumerate(glob.glob(pwd + '/' + filebase)) #0.78, 0.970
        ]
        #Data scaling parameters:
        self.scale_ = np.array([2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
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
        self.mean_ = np.array([ 4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
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

    def predict_instability_time(self, sim, samples=1000, seed=None,
            max_model_samples=100, return_samples=False, prior_above_9=fitted_prior(), Ncpus=None):
        """Estimate instability time for given simulation(s), and the 68% confidence
            interval.

        Returns the median of the posterior, the 16th percentile, and
            the 84th percentile. Uses `samples` samples of the posterior
            to calculate this.

        Parameters:

        sim (rebound.Simulation or list): Orbital configuration(s) to test
        samples (int): Number of samples to use
        seed (int): Random seed
        max_model_samples (int): maximum number of times to re-generate model parameters.
            Larger number increases accuracy but greatly decreases speed.
        return_samples (bool): return the raw samples as a second argument
        prior_above_9 (function): function defining the probability density
            function of instability times above 1e9 orbits of the innermost
            planet. By default is a decaying prior which was fit to the training dataset.
            This takes as input time in terms of the orbits of the innermost planet.
        Ncpus (int): Number of CPUs to use for calculation (only if passing more than one simulation).
            Default: Use all available cpus. 

        Returns:

        center_estimate (float): instability time in units of
            the rebound simulation's time units (e.g., if P=1.
            for the innermost planet, this estimate will be
            in units of orbits)
        lower (float): 16th percentile instability time
        upper (float): 84th percentile instability time
        [t_inst_samples (array): raw samples of the posterior]
        """
        batched = self.is_batched(sim)
        t_inst_samples = self.sample_instability_time(sim,
                samples=samples, seed=seed, max_model_samples=max_model_samples,
                prior_above_9=prior_above_9, Ncpus=Ncpus)
        if batched:
            center_estimate = np.median(t_inst_samples, axis=1)
            upper = np.percentile(t_inst_samples, 100-16, axis=1)
            lower = np.percentile(t_inst_samples,     16, axis=1)
        else:
            center_estimate = np.median(t_inst_samples)
            upper = np.percentile(t_inst_samples, 100-16)
            lower = np.percentile(t_inst_samples,     16)

        if return_samples:
            return center_estimate, lower, upper, t_inst_samples
        else:
            return center_estimate, lower, upper

    def predict_stable(self, sim, tmax=None, samples=1000, seed=None, 
            return_samples=False, max_model_samples=100, prior_above_9=fitted_prior(), Ncpus=None):
        """Estimate chance of stability for given simulation(s).

        Parameters:

        sim (rebound.Simulation or list): Orbital configuration(s) to test
        tmax (float or list): Time at which the system is queried as stable,
            in rebound simulation time units.
        samples (int): Number of samples to use
        seed (int): Random seed
        max_model_samples (int): maximum number of times to re-generate model parameters.
            Larger number increases accuracy but greatly decreases speed.
        return_samples (bool): return the raw samples as a second argument
        prior_above_9 (function): function defining the probability density
            function of instability times above 1e9 orbits of the innermost
            planet. By default is a decaying prior which was fit to the training dataset.
            This takes as input time in terms of the orbits of the innermost planet.
        Ncpus (int): Number of CPUs to use for calculation (only if passing more than one simulation).
            Default: Use all available cpus. 

        Returns:

        p (float): probability of stability past the given tmax
            (default 1e9*min(P) orbits)
        [t_inst_samples (array): raw samples of the posterior]
        """
        batched = self.is_batched(sim)
        t_inst_samples = self.sample_instability_time(sim,
                samples=samples, seed=seed, max_model_samples=max_model_samples,
                prior_above_9=prior_above_9)

        if tmax is None:
            if batched:
                tmax = np.array([
                    1e9 *  
                    np.min([np.abs(p.P) for p in s.particles[1:s.N_real]])
                    for s in sim])
            else:
                minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
                tmax = 1e9 * minP
        elif batched:
            if isinstance(tmax, list):
                tmax = np.array(tmax)
            elif isinstance(tmax, np.ndarray):
                ...
            else:
                tmax = np.ones(len(sim)) * tmax
            assert len(tmax) == len(sim)

        if batched:
            out = np.average(t_inst_samples[:, :] > tmax[:, None], 1)
        else:
            out = np.average(t_inst_samples > tmax)
        
        if return_samples:
            return out, t_inst_samples
        else:
            return out

    def resample_stable_sims(self, samps_time, prior_above_9):
        """Use a prior to re-sample stable instability times"""
        stable_past_9 = samps_time >= 9
        normalization = quad(prior_above_9, a=9, b=np.inf)[0]
        prior = lambda logT: prior_above_9(logT)/normalization
        n_samples = stable_past_9.sum()
        bins = max([10000, n_samples*4])
        top = 100.
        bin_edges = np.linspace(9, top, num=bins)
        cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
        bin_edges = [9.] +list(bin_edges)+[top]
        # Numerically interpolate the inverse cumulative distribution function:
        inv_cdf = interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        t_inst_samples = inv_cdf(r)
        samps_time[stable_past_9] = t_inst_samples
        return samps_time

    def is_batched(self, sim):
        batched = False
        if isinstance(sim, list):
            batched = True
            nsim = len(sim)
            if len(set([s.N_real for s in sim])) != 1:
                raise ValueError("If running over many sims at once, they must have the same number of particles!")
        return batched

    @profile
    def sample_instability_time(self, sim, samples=1000, seed=None,
            max_model_samples=100, prior_above_9=fitted_prior(), Ncpus=None):
        """Return samples from a posterior over instability time for
            given simulation(s). This returns samples from a simple prior for
            all times greater than 10^9 orbits.

        Parameters:

        sim (rebound.Simulation or list): Orbital configuration(s) to test
        samples (int): Number of samples to return
        seed (int): Random seed
        max_model_samples (int): maximum number of times to re-generate model parameters.
            Larger number increases accuracy but greatly decreases speed.
        prior_above_9 (function): function defining the probability density
            function of instability times above 1e9 orbits of the innermost
            planet. By default is a decaying prior which was fit to the training dataset.
            This takes as input time in terms of the orbits of the innermost planet.
        Ncpus (int): Number of CPUs to use for calculation (only if
            passing more than one simulation). Default: Use all available cpus. 

        Returns:

        np.array: samples of the posterior (nsamples,) or (nsim, nsamples) for
            instability time, in units of the rebound simulation.
        """
        batched = self.is_batched(sim)

        if seed is not None:
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.cuda:
                torch.cuda.manual_seed_all(seed)

        if batched:
            n_sims = len(sim)
            if Ncpus is None:
                Ncpus = cpu_count()
            with Pool(Ncpus) as pool:
                pool_out = pool.map(generate_dataset, sim)

            Xs = np.array([X for X in pool_out if isinstance(X, np.ndarray)])
            already_computed_results_idx =   [i for i, X in enumerate(pool_out) if not isinstance(X, np.ndarray)]
            already_computed_results_times = [X for i, X in enumerate(pool_out) if not isinstance(X, np.ndarray)]
        else:
            out = generate_dataset(sim)
            if not isinstance(out, np.ndarray):
                minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
                return np.ones(samples) * out * minP
            Xs = np.array([out])

        if len(Xs) > 0:
            nbatch = Xs.shape[0]
            ntrios = Xs.shape[1]
            nt = 100
            X = E.rearrange(Xs, 'batch trio () time feature -> (batch trio time) feature')
            Xp = (X - self.mean_[None, :]) / self.scale_[None, :]
            Xp = E.rearrange(Xp, '(batch trio time) feature -> (batch trio) time feature',
                             batch=nbatch, trio=ntrios, time=nt)

            Xflat = torch.tensor(Xp).float()
            if self.cuda:
                Xflat = Xflat.cuda()

            model_samples = min([max_model_samples, samples])
            oversample = int(math.ceil(samples/model_samples))
            Xflat = E.repeat(Xflat,
                            '(batch trio) time feature -> (batch trio oversample) time feature',
                            batch=nbatch, trio=ntrios,
                            oversample=oversample)

            sampled_mu_std = np.array([self.sample_full_swag(Xflat).detach().cpu().numpy() for _ in range(model_samples)])
            sampled_mu_std = E.rearrange(sampled_mu_std,
                            'msamples (batch trio oversample) mu_std -> (msamples oversample) (batch trio) mu_std',
                            batch=nbatch, trio=ntrios,
                            oversample=oversample
                            )
            sampled_mu_std = sampled_mu_std[:samples]
            sampled_mu_std = sampled_mu_std.astype(np.float64)
            # print(sampled_mu_std.shape)
            #(samples, (batch trio), mu_std)

            samps_time = np.array(fast_truncnorm(
                    sampled_mu_std[..., 0], sampled_mu_std[..., 1],
                    left=4, d=10000, nsamp=40
                ))
            samps_time = self.resample_stable_sims(samps_time, prior_above_9)
            samps_time = E.rearrange(samps_time,
                             'samples (batch trio) -> batch samples trio',
                             batch=nbatch, trio=ntrios)
            outs = np.min(samps_time, 2)
            time_estimates = np.power(10.0, outs)
            # print(time_estimates.shape)
            #HACK TODO - need already computed estimates

        if not batched:
            minP = np.min([np.abs(p.P) for p in sim.particles[1:sim.N_real]])
            return time_estimates[0] * minP

        j = 0
        k = 0
        correct_order_results = []
        for i in range(n_sims):
            cur_sim = sim[i]
            minP = np.min([np.abs(p.P) for p in cur_sim.particles[1:cur_sim.N_real]])
            if i in already_computed_results_idx:
                correct_order_results.append(already_computed_results_times[k] * np.ones(samples) * minP)
                k += 1
            else:
                correct_order_results.append(time_estimates[j] * minP)
                j += 1

        correct_order_results = np.array(correct_order_results, dtype=np.float64)
        return correct_order_results

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
