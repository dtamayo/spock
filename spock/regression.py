import math
import numpy as np
from scipy.stats import truncnorm
import os
from collections import OrderedDict
from .feature_functions import features
from copy import deepcopy as copy
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.functional import F
from .training_data_functions import restseriesv5, orbtseries



def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))

def soft_clamp(x, lo, high):
    return 0.5*(torch.tanh(x)+1)*(high-lo) + lo

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""
    # From https://github.com/kumar-shridhar/PyTorch-BayesianCNN

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def kl_loss(self):
        kl = 0.0
        for module in self.modules():
            if hasattr(module, '_kl_loss'):
                kl = kl + module._kl_loss()

        return kl

class BBBLinear(ModuleWrapper):
    # From https://github.com/kumar-shridhar/PyTorch-BayesianCNN
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):

        mean = F.linear(x, self.W)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.W * self.W

        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        epsilon = std.data.new(std.size()).normal_()
        # Local reparameterization trick
        out = mean + std * epsilon

        return out

    def _kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()

Linear = BBBLinear
module = ModuleWrapper

def mlp(in_n, out_n, hidden, layers):
    if layers == 0:
        return Linear(in_n, out_n)

    result = [Linear(in_n, hidden),
             act()]
    for i in reversed(range(layers)):
        result.extend([
            Linear(hidden, hidden),
            act()
            ])

    result.extend([nn.Linear(hidden, out_n)])
    return nn.Sequential(*result)

def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x).to(x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)
    
    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) + 
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))
    
    return f_under(x_under) + f_over(x_over)

#Standard settings:
hidden = 100
latent = 30
batch_size = 32
in_length = 1
out_length = 3
act = nn.ReLU
low_sampling_num = 8
log_sample_sampling_num = False
idxes = np.array([0, 2, 3, 5, 7, 8, 12, 13, 16, 19, 22, 23, 25, 27, 28, 29, 30, 32, 33, 34, 37, 38, 40])
axis_mapping = {
        'time': [0],
        'e+': [1+j*8 for j in range(3)],
        'e-': [2+j*8 for j in range(3)],
        'Zcross': [3+j*8 for j in range(3)],
        'phi': [4+j*8 for j in range(3)],
        'Zcom': [5+j*8 for j in range(3)],
        'phiZcom': [6+j*8 for j in range(3)],
        'Zstar': [7+j*8 for j in range(3)],
        'Zfree': [8+j*8 for j in range(3)],
        'AMD': [25],
        'MEGNO': [26],
        'mass': [27, 28, 29],
        # 30 is just time again.
        'e': [31 + i*6 + 0 for i in range(3)],
        'a': [31 + i*6 + 1 for i in range(3)],
        'i': [31 + i*6 + 2 for i in range(3)],
        'Omega': [31 + i*6 + 3 for i in range(3)],
        'pomega': [31 + i*6 + 4 for i in range(3)],
        'M': [31 + i*6 + 5 for i in range(3)]
    }
effective_n_features = len(idxes)

class VarModel(module):
    def __init__(self, hidden=50, latent=5, p=0.1):
        super().__init__()
        self.feature_nn = mlp(effective_n_features, latent, hidden, in_length)
        self.regress_nn = mlp(latent*2, 2, hidden, out_length)
        self.lowest = 0.5
        self.random_sample = True

    def forward(self, x):
        random_sample = self.random_sample
        if random_sample:
            if log_sample_sampling_num:
                samples = int(
                    np.clip(np.exp(np.random.randn()*0.8 + np.log(30)), low_sampling_num, x.shape[1]+1)
                )
            else:
                samples = np.random.randint(low_sampling_num, x.shape[1]+1)
            x = x[:, np.random.randint(0, x.shape[1], size=samples)]
        
        x = self.feature_nn(x)
        sample_mu = torch.mean(x, dim=1)
        sample_var = torch.std(x, dim=1)**2
        n = x.shape[1]
        
        std_in_mu = torch.sqrt(sample_var/n)
        std_in_var = torch.sqrt(2*sample_var**2/(n-1))
        
        mu_sample = torch.randn_like(sample_mu )*std_in_mu + sample_mu
        var_sample = torch.randn_like(sample_var)*std_in_var + sample_var
        
        clatent = torch.cat((mu_sample, var_sample), dim=1)
        
        testy = self.regress_nn(clatent)
        mu = soft_clamp(testy[:, [0]], 4.0, 12.0)
        std = soft_clamp(testy[:, [1]], self.lowest, 6.0)
        return torch.cat((mu, std), dim=1)

class StabilityRegression(object):
    def __init__(self):
        super(StabilityRegression, self).__init__()
        pwd = os.path.dirname(__file__)
        model_state = torch.load(open(pwd + '/models/feb23_best_modelv6_bbb.pt', 'rb'))
        self.model = VarModel(hidden, latent=latent)
        self.model.load_state_dict(model_state['state'])
        self.mean_ = model_state['mean']
        self.scale_ = model_state['scale']
        assert np.all(idxes == model_state['idxes'])

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
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets") 
        if indices:
            if len(indices) != 3:
                raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
            trios = [indices] # always make it into a list of trios to test
        else:
            trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios
        
        #Simplicity for start:
        all_full_samples = []
        simt = sim.copy()
        for trio in trios:
            sim = simt.copy()

            kwargs = OrderedDict()
            mult = 1
            kwargs['Norbits'] = 1e4 * mult
            kwargs['Nout'] = 1000 * mult
            kwargs['window'] = 10
            args = list(kwargs.values())
            # These are the .npy.
            # In the other file, we concatenate (restseries, orbtseries, mass_array)
            restseries_array = restseriesv5(sim, args, trio=trio)
            orbtseries_array = orbtseries(sim, args, trio=trio)
            mass_array = np.array([sim.particles[i].m/sim.particles[0].m for i in range(1, 4)])
            mass_array = np.tile(mass_array[None], (1000, 1))
            together = np.concatenate((restseries_array, orbtseries_array, mass_array), axis=1)
            relevant_axes=['e',
                'a',
                'e-',
                'e+',
                'AMD',
                'MEGNO',
                'mass',
                'pomega',
                'Omega',
                'Zcross',
                'phi',
                'phiZcom',
                'Zstar',
                'Zfree',
                'M'
            ]
            flatten = lambda l: [subitem for item in l for subitem in item]
            axes_to_take = sorted(flatten([axis_mapping[axis_name] for axis_name in relevant_axes]))
            prepared = together[None, ::10, axes_to_take][..., idxes]
            prepared = (prepared - self.mean_[None, None]) / self.scale_[None, None]
            prepared = np.tile(prepared, (samples, 1, 1))
            prepared = torch.from_numpy(prepared).float()

            out = np.concatenate([
                self.model(prepared[slic]).detach().numpy()
                for slic in np.array_split(
                    np.arange(len(prepared)),
                    len(prepared)//5000
                    )])
            out = out.reshape(-1, samples, 2)
            a, b = (4 - out[..., 0]) / out[..., 1], (12 - out[..., 0]) / out[..., 1]
            full_samples = truncnorm.rvs(a, b, loc=out[..., 0], scale=out[..., 1])
            all_full_samples.append(full_samples)

        out = np.array(all_full_samples)
        best_out_over_trios = out[np.argsort(np.average(out, axis=1))[0]]
        return 10**best_out_over_trios

