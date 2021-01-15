import pickle as pkl
from copy import deepcopy as copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib as mpl
mpl.use('agg')
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import sys
import torch.nn.functional as F
from torch.nn import Parameter
import math
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import math
from torch._six import inf
from functools import wraps
import warnings
from torch.optim.optimizer import Optimizer
from collections import OrderedDict


class CustomOneCycleLR(torch.optim.lr_scheduler._LRScheduler):
    """Custom version of one-cycle learning rate to stop early"""
    def __init__(self,
                 optimizer,
                 max_lr,
                 swa_steps_start,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        total_steps = swa_steps_start #Just fix afterwards.
        epochs = None
        steps_per_epoch = None
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected non-negative integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(CustomOneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        if pct >= 1.0:
            return end
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        if pct >= 1.0:
            return end
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        lrs = []
        step_num = self.last_epoch
        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))
        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)
            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum
        return lrs

def get_data(
        ssX=None,
        batch_size=32,
        train=True,
        **kwargs):
    """
    inputs:
        batch_size: int

    return:
        (dataloader, test_dataloader)
    """
    plot_random = False if 'plot_random' not in kwargs else kwargs['plot_random']
    plot_resonant = not plot_random
    train_all = False if 'train_all' not in kwargs else kwargs['train_all']
    plot = False if 'plot' not in kwargs else kwargs['plot']
    if not train_all and ssX is None:
        plot_resonant = True
        plot_random = False

    if train_all:
        filename = 'data/combined.pkl'
    elif plot_resonant:
        filename = 'data/resonant_dataset.pkl'
    elif plot_random:
        filename = 'data/random_dataset.pkl'

    # These are generated by data_from_pkl.py
    loaded_data = pkl.load(
        open(filename, 'rb')
    )

    train_ssX = (ssX is None)

    fullX, fully = loaded_data['X'], loaded_data['y']

    if train_all:
        len_random = 17082 #Number of valid random examples (others have NaNs)
        random_data = np.arange(len(fullX)) >= (len(fullX) - len_random)


    # Differentiate megno
    if 'fix_megno' in kwargs and kwargs['fix_megno']:
        idx = [i for i, lab in enumerate(loaded_data['labels']) if 'megno' in lab][0]
        fullX[:, 1:, idx] -= fullX[:, :-1, idx]

    if 'include_derivatives' in kwargs and kwargs['include_derivatives']:
        derivative = fullX[:, 1:, :] - fullX[:, :-1, :]
        derivative = np.concatenate((
            derivative[:, [0], :],
            derivative), axis=1)
        fullX = np.concatenate((
            fullX, derivative),
            axis=2)


    # Hide fraction of test
    # MAKE SURE WE DO COPIES AFTER!!!!
    if train:
        if train_all:
            remy, finaly, remX, finalX, rem_random, final_random = train_test_split(fully, fullX, random_data, shuffle=True, test_size=1./10, random_state=0)
            trainy, testy, trainX, testX, train_random, test_random = train_test_split(remy, remX, rem_random, shuffle=True, test_size=1./10, random_state=1)
        else:
            remy, finaly, remX, finalX = train_test_split(fully, fullX, shuffle=True, test_size=1./10, random_state=0)
            trainy, testy, trainX, testX = train_test_split(remy, remX, shuffle=True, test_size=1./10, random_state=1)
    else:
        assert not train_all
        remy = fully
        finaly = fully
        testy = fully
        trainy = fully
        remX = fullX
        finalX = fullX
        testX = fullX
        trainX = fullX

    if plot:
        # Use test dataset for plotting, so put it in validation part:
        testX = finalX
        testy = finaly

    if train_ssX:
        if 'power_transform' in kwargs and kwargs['power_transform']:
            ssX = PowerTransformer(method='yeo-johnson') #Power is best
        else:
            ssX = StandardScaler() #Power is best

    n_t = trainX.shape[1]
    n_features = trainX.shape[2]

    if train_ssX:
        ssX.fit(trainX.reshape(-1, n_features)[::1539])

    ttrainy = trainy
    ttesty = testy
    ttrainX = ssX.transform(trainX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    ttestX = ssX.transform(testX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    if train_all:
        ttest_random = test_random
        ttrain_random = train_random

    tremX = ssX.transform(remX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    tremy = remy

    train_len = ttrainX.shape[0]
    X = Variable(torch.from_numpy(np.concatenate((ttrainX, ttestX))).type(torch.FloatTensor))
    y = Variable(torch.from_numpy(np.concatenate((ttrainy, ttesty))).type(torch.FloatTensor))
    if train_all:
        r = Variable(torch.from_numpy(np.concatenate((ttrain_random, ttest_random))).type(torch.BoolTensor))

    Xrem = Variable(torch.from_numpy(tremX).type(torch.FloatTensor))
    yrem = Variable(torch.from_numpy(tremy).type(torch.FloatTensor))

    idxes = np.s_[:]
    dataset = torch.utils.data.TensorDataset(X[:train_len, :, idxes], y[:train_len])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    # Cut up dataset into only the random or resonant parts. 
    # Only needed if plotting OR 
    if (not plot) or (not train_all):
        test_dataset = torch.utils.data.TensorDataset(X[train_len:, :, idxes], y[train_len:])
    else:
        if plot_random: mask =  r
        else:           mask = ~r
        print(f'Plotting with {mask.sum()} total elements, when plot_random={plot_random}')
        test_dataset = torch.utils.data.TensorDataset(X[train_len:][r[train_len:]][:, :, idxes], y[train_len:][r[train_len:]])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False, pin_memory=True, num_workers=8)
        
    kwargs['model'].ssX = copy(ssX)

    return dataloader, test_dataloader


def soft_clamp(x, lo, high):
    return 0.5*(torch.tanh(x)+1)*(high-lo) + lo

Linear = nn.Linear
module = nn.Module

def mlp(in_n, out_n, hidden, layers, act='relu'):
    if act == 'relu':
        act = nn.ReLU
    elif act == 'softplus':
        act = nn.Softplus
    else:
        raise NotImplementedError('act must be relu or softplus')

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
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)
    
    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) + 
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))
    
    return f_under(x_under) + f_over(x_over)

EPSILON = 1e-5

class VarModel(pl.LightningModule):
    """Bayesian Neural Network model for predicting instability time"""
    def __init__(self, hparams):
        super().__init__()
        if 'seed' not in hparams: hparams['seed'] = 0
        # pl.seed_everything(hparams['seed'])

        hparams['include_derivatives'] = False if 'include_derivatives' not in hparams else hparams['include_derivatives']

        if 'time_series_features' not in hparams:
            hparams['time_series_features'] = 38+3

        if hparams['time_series_features'] == 82:
            hparams['time_series_features'] = 41

        self.fix_megno = False if 'fix_megno' not in hparams else hparams['fix_megno']
        self.fix_megno2 = False if 'fix_megno2' not in hparams else hparams['fix_megno2']
        self.include_angles = False if 'include_angles' not in hparams else hparams['include_angles']

        self.n_features = hparams['time_series_features'] * (1 + int(hparams['include_derivatives']))
        self.feature_nn = mlp(self.n_features, hparams['latent'], hparams['hidden'], hparams['in'])
        self.regress_nn = mlp(hparams['latent']*2 + int(self.fix_megno)*2, 2, hparams['hidden'], hparams['out'])
        self.input_noise_logvar = nn.Parameter(torch.zeros(self.n_features)-2)
        self.summary_noise_logvar = nn.Parameter(torch.zeros(hparams['latent'] * 2 + int(self.fix_megno)*2) - 2) # add to summaries, not direct latents
        self.lowest = 0.5
        if 'lower_std' in hparams and hparams['lower_std']:
            self.lowest = 0.1

        self.latents = None
        self.beta_in = 1 if 'beta_in' not in hparams else hparams['beta_in']
        self.beta_out = 1 if 'beta_out' not in hparams else hparams['beta_out']
        self.megno_location = 7
        self.mmr_location = [3, 6]
        self.nan_location = [38, 39, 40]
        self.eplusminus_location = [1, 2, 4, 5]

        # SWA params
        hparams['scheduler_choice'] = 'swa' #'cycle' if 'scheduler_choice' not in hparams else hparams['scheduler_choice']
        hparams['save_freq'] = 25 if 'save_freq' not in hparams else hparams['save_freq']
        hparams['eval_freq'] = 5 if 'eval_freq' not in hparams else hparams['eval_freq']
        hparams['momentum'] = 0.9 if 'momentum' not in hparams else hparams['momentum']
        hparams['weight_decay'] = 1e-4 if 'weight_decay' not in hparams else hparams['weight_decay']
        hparams['noisy_val'] = True if 'noisy_val' not in hparams else hparams['noisy_val']

        # self.hparams = hparams
        self.save_hyperparameters()
        self.steps = hparams['steps']
        self.batch_size = hparams['batch_size']
        self.lr = hparams['lr'] #init_lr
        self._dataloader = None
        self._val_dataloader = None
        self.random_sample = False if 'random_sample' not in hparams else hparams['random_sample']
        self.train_len = 78660
        self.test_len = 8740
        self._summary_kl = 0.0
        self.include_mmr = hparams['include_mmr']
        self.include_nan = hparams['include_nan']
        self.include_eplusminus = True if 'include_eplusminus' not in hparams else hparams['include_eplusminus']
        self.train_all = False if 'train_all' not in hparams else hparams['train_all']

        self._cur_summary = None

        self.ssX = None
        self.ssy = None

    def augment(self, x):
        # This randomly samples times.
        samples = np.random.randint(self.hparams['samp'], x.shape[1]+1)
        x = x[:, np.random.randint(0, x.shape[1], size=samples)]
        return x

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def compute_summary_stats(self, x):
        x = self.feature_nn(x)
        sample_mu = torch.mean(x, dim=1)
        sample_var = torch.std(x, dim=1)**2
        n = x.shape[1]
        
        std_in_mu = torch.sqrt(sample_var/n)
        std_in_var = torch.sqrt(2*sample_var**2/(n-1))
        
        # Take a "sample" of the average/variance of the learned features
        mu_sample =  torch.randn_like(sample_mu) *std_in_mu  + sample_mu
        var_sample = torch.randn_like(sample_var)*std_in_var + sample_var

        # Get to same unit
        std_sample = torch.sqrt(torch.abs(var_sample) + EPSILON)
        #clatent = torch.cat((mu_sample, var_sample), dim=1)
        clatent = torch.cat((mu_sample, std_sample), dim=1)
        self.latents = x

        return clatent

    def predict_instability(self, summary_stats):
        testy = self.regress_nn(summary_stats)
        # Outputs mu, std
        mu = soft_clamp(testy[:, [0]], 4.0, 12.0)
        std = soft_clamp(testy[:, [1]], self.lowest, 6.0)
        return mu, std

    def add_input_noise(self, x):
        noise = torch.randn_like(x, device=self.device) * torch.exp(self.input_noise_logvar[None, None, :]/2)
        return x + noise

    def add_summary_noise(self, summary_stats):
        noise = torch.randn_like(summary_stats, device=self.device) * torch.exp(self.summary_noise_logvar[None, :]/2)
        return summary_stats + noise

    def zero_megno(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.megno_location] = x[..., self.megno_location].clone()
            x = x - mask
        return x

    def zero_mmr(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.mmr_location] = x[..., self.mmr_location].clone()
            x = x - mask
        return x

    def zero_nan(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.nan_location] = x[..., self.nan_location].clone()
            x = x - mask
        return x

    def zero_eplusminus(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.eplusminus_location] = x[..., self.eplusminus_location].clone()
            x = x - mask
        return x

    def summarize_megno(self, x):
        megno_avg = torch.mean(x[:, :, [self.megno_location]], 1)
        megno_std = torch.std(x[:, :, [self.megno_location]], 1)

        return torch.cat([megno_avg, megno_std], dim=1)

    def forward(self, x, noisy_val=True):
        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        if self.random_sample:
            x = self.augment(x)
        #x is (batch, time, feature)
        if noisy_val:
            x = self.add_input_noise(x)
        
        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        self._cur_summary = summary_stats

        #summary is (batch, feature)
        self._summary_kl = (1/2) * (
                summary_stats**2
                + torch.exp(self.summary_noise_logvar)[None, :]
                - self.summary_noise_logvar[None, :]
                - 1
            )

        if noisy_val:
            summary_stats = self.add_summary_noise(summary_stats)

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)

    def sample(self, x, samples=10):
        all_samp = []
        init_settings = [self.random_sample, self.device]
        self.cpu()
        x = x.cpu()
        self.random_sample = False
        for _ in range(samples):
            out = self(x).detach().numpy()
            mu = out[:, 0]
            std = out[:, 1]
            all_samp.append(
                mu + np.random.randn(len(out))*std
            )
        self.random_sample = init_settings[0]
        self.to(init_settings[1])
        return np.average(all_samp, axis=0) 

    def _lossfnc(self, testy, y):
        mu = testy[:, [0]]
        std = testy[:, [1]]

        var = std**2
        t_greater_9 = y >= 9

        regression_loss = -(y - mu)**2/(2*var)
        regression_loss += -torch.log(std)
        regression_loss += -safe_log_erf(
                    (mu - 4)/(torch.sqrt(2*var))
                )
        classifier_loss = safe_log_erf(
                    (mu - 9)/(torch.sqrt(2*var))
            )

        safe_regression_loss = torch.where(
                ~torch.isfinite(regression_loss),
                -torch.ones_like(regression_loss)*100,
                regression_loss)
        safe_classifier_loss = torch.where(
                ~torch.isfinite(classifier_loss),
                -torch.ones_like(classifier_loss)*100,
                classifier_loss)

        total_loss = (
            safe_regression_loss * (~t_greater_9) +
            safe_classifier_loss * ( t_greater_9)
        )

        return -total_loss.sum(1)

    def lossfnc(self, x, y, samples=1, noisy_val=True):
        testy = self(x, noisy_val=noisy_val)
        n_samp = y.shape[0]
        loss = self._lossfnc(testy, y).sum()
        return loss

    def input_kl(self):
        return (1/2) * (
                torch.exp(self.input_noise_logvar)
                - self.input_noise_logvar
                - 1
            ).sum()

    def summary_kl(self):
        return self._summary_kl.sum()

    def training_step(self, batch, batch_idx):
        fraction = self.global_step / self.hparams['steps']
        beta_in = min([1, fraction/0.3]) * self.beta_in
        beta_out = min([1, fraction/0.3]) * self.beta_out

        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        #cur_frac = len(X_sample) / self.train_len

        # Want to be important with total number of samples
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out

        prior = input_kl + summary_kl

        total_loss = loss + prior

        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample), 'train_loss_with_reg': total_loss/len(X_sample), 'input_kl': input_kl/len(X_sample), 'summary_kl': summary_kl/len(X_sample)}

        return {'loss': total_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()

        tensorboard_logs = {'val_loss_no_reg': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])

        assert self.hparams['scheduler_choice'] == 'swa'
        scheduler = CustomOneCycleLR(opt1, self.lr, int(0.9*self.steps), final_div_factor=1e4)
        interval = 'steps'
        name = 'swa_lr'

        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def make_dataloaders(self, train=True, **extra_kwargs):
        kwargs = {
            **self.hparams,
            'model': self,
            **extra_kwargs,
            'train': train,
        }
        if 'ssX' in kwargs:
            dataloader, val_dataloader = get_data(**kwargs)
        else:
            dataloader, val_dataloader = get_data(ssX=self.ssX, **kwargs)

        labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']
        for i in range(len(labels)):
            label = labels[i]
            if not ('cos' in label or
                'sin' in label or
                'nan_' in label or
                label == 'i1' or
                label == 'i2' or
                label == 'i3'):
                continue

            if not self.include_angles:
                print('Tossing', i, label)
                dataloader.dataset.tensors[0][..., i] = 0.0
                val_dataloader.dataset.tensors[0][..., i] = 0.0

        self._dataloader = dataloader
        self._val_dataloader = val_dataloader
        self.train_len = len(dataloader.dataset.tensors[0])
        self.test_len = len(val_dataloader.dataset.tensors[0])

    def train_dataloader(self):
        if self._dataloader is None:
            self.make_dataloaders()
        return self._dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self.make_dataloaders()
        return self._val_dataloader


class SWAGModel(VarModel):
    """Use .load_from_checkpoint(checkpoint_path) to initialize a SWAG model"""
    def init_params(self, swa_params):
        self.swa_params = swa_params
        self.swa_params['swa_lr'] = 0.001 if 'swa_lr' not in self.swa_params else self.swa_params['swa_lr']
        self.swa_params['swa_start'] = 1000 if 'swa_start' not in self.swa_params else self.swa_params['swa_start']
        self.swa_params['swa_recording_lr_factor'] = 0.5 if 'swa_recording_lr_factor' not in self.swa_params else self.swa_params['swa_recording_lr_factor']

        self.n_models = 0
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.K = 20 if 'K' not in self.swa_params else self.swa_params['K']
        self.c = 2 if 'c' not in self.swa_params else self.swa_params['c']
        self.swa_params['c'] = self.c
        self.swa_params['K'] = self.K

        return self

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.swa_params['swa_lr'], momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, [self.swa_params['swa_start']], self.swa_params['swa_recording_lr_factor'])
        interval = 'steps'
        name = 'swa_record_lr'
        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def training_step(self, batch, batch_idx):
        beta_in = self.beta_in
        beta_out = self.beta_out
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out
        prior = input_kl + summary_kl
        total_loss = loss + prior
        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample), 'train_loss_with_reg': total_loss/len(X_sample), 'input_kl': input_kl/len(X_sample), 'summary_kl': summary_kl/len(X_sample)}
        return {'loss': total_loss, 'log': tensorboard_logs}

    def flatten(self):
        """Convert state dict into a vector"""
        ps = self.state_dict()
        p_vec = None
        for key in ps.keys():
            p = ps[key]

            if p_vec is None:
                p_vec = p.reshape(-1)
            else:
                p_vec = torch.cat((p_vec, p.reshape(-1)))

        return p_vec

    def load(self, p_vec):
        """Load a vector into the state dict"""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key in cur_state_dict.keys():
            old_p = cur_state_dict[key]
            size = old_p.numel()
            shape = old_p.shape
            new_p = p_vec[i:i+size].reshape(*shape)
            new_state_dict[key] = new_p
            i += size

        self.load_state_dict(new_state_dict)

    def aggregate_model(self):
        """Aggregate models for SWA/SWAG"""

        cur_w = self.flatten()
        cur_w2 = cur_w ** 2
        with torch.no_grad():
            if self.w_avg is None:
                self.w_avg = cur_w
                self.w2_avg = cur_w2
            else:
                self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
                self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

            if self.pre_D is None:
                self.pre_D = cur_w.clone()[:, None]
            elif self.current_epoch % self.c == 0:
                #Record weights, measure discrepancy with average later
                self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
                if self.pre_D.shape[1] > self.K:
                    self.pre_D = self.pre_D[:, 1:]
                    

        self.n_models += 1

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len

        if self.w_avg is None:
            swa_loss = loss
        else:
            tmp = self.flatten()
            self.load(self.w_avg)
            swa_loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len
            self.load(tmp)

        return {'val_loss': loss, 'swa_loss': swa_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        swa_avg_loss = torch.stack([x['swa_loss'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss_no_reg': avg_loss, 'swa_loss_no_reg': swa_avg_loss}
        #TODO: Check
        #fraction = self.global_step / self.hparams['steps']
        #if fraction > 0.5:
        if self.global_step > self.hparams['swa_start']:

            self.aggregate_model()

        # Record validation loss, and aggregated model loss
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_weights(self, scale=1):
        """Sample weights using SWAG:
        - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
            - This can be done with the following matrices:
                - z_1 ~ N(0, I_d); d the number of parameters
                - z_2 ~ N(0, I_K)
            - Then, compute:
            - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        """
        with torch.no_grad():
            avg_w = self.w_avg #[K]
            avg_w2 = self.w2_avg #[K]
            D = self.pre_D - avg_w[:, None]#[d, K]
            d = avg_w.shape[0]
            K = self.K
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((K, 1), device=self.device)
            sigma = torch.abs(torch.diag(avg_w2 - avg_w**2))

            w = avg_w[None] + scale * (1.0/np.sqrt(2.0)) * z_1 @ sigma**0.5
            w += scale * (D @ z_2).T / np.sqrt(2*(K-1))
            w = w[0]

        self.load(w)

    def forward_swag(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)
        self._summary_kl = (1/2) * (
                summary_stats**2
                + torch.exp(self.summary_noise_logvar)[None, :]
                - self.summary_noise_logvar[None, :]
                - 1
            )

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)

    def forward_swag_fast(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)


def save_swag(swag_model, path):
    save_items = {
        'hparams':swag_model.hparams,
        'swa_params': swag_model.swa_params,
        'w_avg': swag_model.w_avg.cpu(),
        'w2_avg': swag_model.w2_avg.cpu(),
        'pre_D': swag_model.pre_D.cpu()
    }

    torch.save(save_items, path)
    
def load_swag(path):
    save_items = torch.load(path)
    swag_model = (
        SWAGModel(save_items['hparams'])
        .init_params(save_items['swa_params'])
    )
    swag_model.w_avg = save_items['w_avg']
    swag_model.w2_avg = save_items['w2_avg']
    swag_model.pre_D = save_items['pre_D']
    
    return swag_model
