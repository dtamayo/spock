import numpy as np
import pandas as pd
import sys
import torch
import torch.nn
import torch.autograd as autograd
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
import torch
import torch.nn as nn
from icecream import ic
import warnings
warnings.filterwarnings('ignore')
dtype = torch.FloatTensor
long_dtype = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn.functional as F
from icecream import ic
import sys
from filenames import cdataset as data_directories
from tqdm import tqdm

NUMCHAN=8
perturb_y_data = False
TOP = 9

def gen_dens(y, bins=int(5.0/0.5*10)):
    dens, bins = np.histogram(y[:, 0].ravel(), bins=bins, density=True, range=[4, TOP])
    dens[dens == 0.] = np.min(dens[dens > 0.])
    plt.clf()
    inv_dens = 1.0/dens
    inv_dens /= np.average(inv_dens)
    bins = bins
    return inv_dens, bins

def load_data_normalized(debug=False, downsample=10,
        dataset='resonant'):

    base = './data/summary_features/'
    labels = []
    mass_ratios = []
    time_series = []
    # features = []
    if dataset == 'resonant':
        from filenames import cdataset as data_directories
    elif dataset == 'random':
        from filenames import cdataset_rand as data_directories
    elif dataset == 'combined':
        from filenames import cdataset_rand, cdataset
        data_directories = cdataset + cdataset_rand
    else:
        raise NotImplementedError

    from icecream import ic

    total_num = []

    for dataset in tqdm(data_directories):

        dataset_base = base + dataset + '/get_extended_tseriesNorbits10000.0Nout1000trio/'
        # features_base = base + dataset + '/resparamsv5Norbits10000.0Nout1000window10/'
        try:
            time_series_tmp = np.load(dataset_base + 'trainingdata.npy', allow_pickle=True)[:, ::downsample]
            assert time_series_tmp.shape[1] == 100
            labels_tmp = pd.read_csv(dataset_base + 'labels.csv')
            mass_ratios_tmp = pd.read_csv(dataset_base + 'massratios.csv')
            # features_tmp = pd.read_csv(features_base + 'trainingdata.csv')
        except (FileNotFoundError, IndexError):
            print('Skipping', dataset)
            continue

        time_series.append(time_series_tmp)
        labels.append(labels_tmp)
        mass_ratios.append(mass_ratios_tmp)
        # features.append(features_tmp)
        total_num.append(len(labels_tmp))
        ic(total_num[-1])

        if dataset[:4] == 'only':
            labels[-1]['instability_time'] = 1e9
            labels[-1]['shadow_instability_time'] = 1e9
        if debug:
            break

    time_series = np.concatenate(time_series)
    mass_ratios = pd.concat(mass_ratios)
    labels = pd.concat(labels)
    ic(len(labels))
    last_dataset = (np.arange(len(labels)) >= np.cumsum(total_num)[-2])
    ic(last_dataset.sum())

    mass_array = np.transpose(np.tile(np.array(mass_ratios[['m1', 'm2', 'm3']]), (1000//downsample, 1, 1)), [1, 0, 2])
    old_axis_labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'Omega1', 'pomega1', 'theta1', 'a2', 'e2', 'i2', 'Omega2', 'pomega2', 'theta2', 'a3', 'e3', 'i3', 'Omega3', 'pomega3', 'theta3']

    old_X = np.concatenate((time_series, mass_array), axis=2)

    old_axis_labels.extend(['m1', 'm2', 'm3'])
    y = np.array(np.log10(labels[['instability_time', 'shadow_instability_time']])).astype(np.float32)


    ic('Removing rows with time as NaN')
    tmp_good_rows = ~np.any(~np.isfinite(old_X)[:, :, [0]], axis=(1, 2))
    tmp_good_rows = tmp_good_rows * (~np.any(y <= 4, axis=1))
    old_X = old_X[tmp_good_rows]
    y = y[tmp_good_rows]
    last_dataset = last_dataset[tmp_good_rows]
    ic('Done')

    ic('Last dataset is', last_dataset.sum(), 'in size')
    ic('Last dataset is', tmp_good_rows[np.cumsum(total_num)[-2]:].sum(), 'in size')

    # print(old_X.shape, len(old_axis_labels))
    ic('Converting nan to 0 in variables, and making new labels for nans')
    isnotfinite = lambda _x: ~np.isfinite(_x)

    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [3]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [6]]).astype(np.float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [7]]).astype(np.float)), axis=2)

    old_axis_labels.extend(['nan_mmr_near', 'nan_mmr_far', 'nan_megno'])

    old_X[:, :, [3, 6, 7]] = np.nan_to_num(old_X[..., [3, 6, 7]], posinf=0.0, neginf=0.0)
    ic('Done')

    axis_labels = []
    X = []

    ic('Cosining')
    for i, label in enumerate(old_axis_labels):
        if 'Omega' in label or 'pomega' in label or 'theta' in label:
            X.append(np.cos(old_X[:, :, [i]]))
            X.append(np.sin(old_X[:, :, [i]]))
            axis_labels.append('cos_'+label)
            axis_labels.append('sin_'+label)
        else:
            X.append(old_X[:, :, [i]])
            axis_labels.append(label)

    X = np.concatenate(X, axis=2)
    ic('Done cosining')

    return {'X': X, 'y': y, 'labels': axis_labels}

globalX = None
globalX_easy = None
globaly = None
globaldens = None
globalbins = None

def reweighted_loss_func(y, yp):
    global globaldens
    global globalbins

    best_time_bin_idx = np.argmin(np.abs(globalbins[np.newaxis, :-1] - y.ravel()[:, np.newaxis]), 1)
    corresponding_weight = globaldens[best_time_bin_idx]

    return np.average(np.square(y.ravel() - yp.ravel()), weights=corresponding_weight.ravel())

def get_accuracy_for_model(
        clf, transform=False, cv=3, requires_flatten=False,
        train_on_shadow_time=False,
        train_on_features=True, train_on_series=False,
        leave_early=False, torch=True,
        add_time_axis=False, loss_func=None,
        reset_data=False,
        **kwargs):
    """[deprecated] Function for getting accuracy of an sklearn-like model."""

    from sklearn.model_selection import KFold
    if loss_func is None:
        from sklearn.metrics import mean_squared_error as loss_func

    global globalX
    global globalX_easy
    global globaly
    global globaldens
    global globalbins

    ic("Loading data")
    if globalX is None or reset_data:
        data = load_data(**kwargs)
        globalX, globalX_easy, globaly = data['X'], data['X_easy'], data['y']
        globaldens = data['density']
        globalbins = data['density_grid']
    ic("Loaded")

    X = globalX
    X_easy = globalX_easy
    y = globaly
    dens = globaldens
    bins = globalbins

    #Remove test data
    N = len(X)
    saveN = int(4.*N/5)
    X, saveX = X[:saveN], X[saveN:]
    X_easy, saveX_easy = X_easy[:saveN], X_easy[saveN:]
    y, savey = y[:saveN], y[saveN:]

    ic("Transforming")
    if transform is not None:
        X_easy = transform.fit_transform(X_easy)

    if not train_on_shadow_time:
        y = np.average(y, axis=1).reshape(-1, 1)
    ic("Transformed")
    loss_scores = []

    fold = KFold(cv, shuffle=False)
    for train_idx, test_idx in fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        X_easy_train, X_easy_test = X_easy[train_idx], X_easy[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if requires_flatten:
            X_train = X_train.reshape(len(X_train), -1)
            X_test = X_test.reshape(len(X_test), -1)

        if train_on_features and not train_on_series:
            clf.fit(X_easy_train, y_train)
            cX = X_easy_test
            times = clf.predict(X_easy_test)
        elif train_on_series and not train_on_features:
            clf.fit(X_train, y_train)
            cX = X_test
            times = clf.predict(X_test)
        else:
            clf.fit(X_train, X_easy_train, y_train)
            cX = X_test
            times = clf.predict(X_test, X_easy_test)
        if torch:
            loss_scores.append(loss_func(times, y_test).cpu().numpy())
        else:
            loss_scores.append(loss_func(times, y_test))
        if leave_early:
            break

    return np.average(loss_scores), clf, transform, (cX, y_test, times), X_easy_test

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

class TorchWrapper(object):

    def __init__(self, clf, run_reshape=None, epochs=100, custom_loss=lambda _: 0,
            post_init=lambda _: 0, l2_reg=1e-5, lr=1e-3, batch_size=32, preprocess=None,
            val_patience=100, patience=300, factor=0.3, classifier=None, train_on_fraction=1.0,
            augment_data_during_predict=False, no_weight_init=False,
            augment_data=False, **kwargs):
        super().__init__()
        self.clf = clf
        self._cuda = False
        if run_reshape is None:
            run_reshape = lambda _: _
        self.run_reshape = run_reshape
        self.epochs = epochs
        self.custom_loss = custom_loss
        self.post_init = post_init
        self.l2_reg = l2_reg
        self.lr = lr
        self.train_on_fraction = train_on_fraction
        self.batch_size = batch_size
        self.augment_data_during_predict = augment_data_during_predict
        self.patience = patience
        self.val_patience = val_patience
        self.factor = factor
        if preprocess is None:
            preprocess = lambda _: _
        self.preprocess = preprocess
        self.classifier = classifier
        self.augment_data = augment_data
        self.no_weight_init = no_weight_init
        self.reset()

    def reset(self):
        if self.no_weight_init:
            return 
        self.clf.cpu()
        self.clf.apply(weight_init)
        self.post_init(self)
        if self._cuda:
            self.clf.cuda()

    def cuda(self):
        self.clf.cuda()
        self._cuda = True
        return self

    def fit(self, X, X_easy, y):
        self.reset()
        self.clf.train()

        ic("Preprocessing")
        X = self.preprocess(X)
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        X_easy = Variable(torch.from_numpy(X_easy).type(torch.FloatTensor))
        y = Variable(torch.from_numpy(y).type(torch.FloatTensor))

        N = len(X)
        train_len = N*4//5
        dataset = torch.utils.data.TensorDataset(X[:train_len], X_easy[:train_len], y[:train_len])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = torch.utils.data.TensorDataset(X[train_len:], X_easy[train_len:], y[train_len:])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.classifier is None:
            criterion_classifier = nn.MSELoss(reduction='mean')
        else:
            criterion_classifier = self.classifier

        best_state_dict = None
        starter_epoch = 0
        best_val_loss = np.inf
        lr = self.lr
        val_patience = self.val_patience

        while starter_epoch < self.epochs - 1:
            if starter_epoch > 0:
                # Second time + through!
                lr *= 0.8

            #solver_classifier = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=10**(-13*np.random.random()))
            solver_classifier = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=self.l2_reg)
            val_bad_count = 0

            ic("Starting epochs")
            for epoch in tqdm(range(starter_epoch, self.epochs), file=sys.stdout):

                self.clf.train()
                losses = 0
                val_losses = 0
                samples_viewed = 0
                samples_to_take = self.train_on_fraction * train_len
                for X_sample, X_easy_sample, y_sample in dataloader:
                    if self._cuda:
                        X_sample = X_sample.cuda()
                        X_easy_sample = X_easy_sample.cuda()
                        y_sample = y_sample.cuda()

                    if self.augment_data:
                        # TODO: Take out time stamp!
                        X_sample = self.augment_data(X_sample)

                    X_sample = self.run_reshape(X_sample).detach()
                    y_sample_pred = self.clf(X_sample, X_easy_sample)
                    loss = criterion_classifier(y_sample_pred, y_sample)
                    loss += self.custom_loss(self)
                    loss.backward()
                    losses += loss.item() / self.train_on_fraction
                    solver_classifier.step()
                    solver_classifier.zero_grad()

                    samples_viewed += self.batch_size
                    if samples_viewed > samples_to_take:
                        break


                # Get validation loss
                self.clf.eval()
                for X_sample, X_easy_sample, y_sample in test_dataloader:
                    if self._cuda:
                        X_sample = X_sample.cuda()
                        X_easy_sample = X_easy_sample.cuda()
                        y_sample = y_sample.cuda()
                    if self.augment_data:
                        # TODO: Take out time stamp!
                        X_sample = self.augment_data(X_sample)
                    X_sample = self.run_reshape(X_sample).detach()
                    y_sample_pred = self.clf(X_sample, X_easy_sample)
                    loss = criterion_classifier(y_sample_pred, y_sample)
                    loss += self.custom_loss(self)
                    val_losses += loss.item()

                avg_loss = losses / train_len
                avg_val_loss = val_losses / (len(X) - train_len)
                print('epoch, avg, val:', epoch, avg_loss, avg_val_loss, flush=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    from copy import deepcopy as copy
                    best_state_dict = copy(self.clf.state_dict())
                    #ic()
                    #ic(best_val_loss, epoch, starter_epoch, avg_val_loss, val_bad_count, lr)
                    val_bad_count = 0
                else:
                    #ic()
                    #ic(best_val_loss, epoch, starter_epoch, avg_val_loss, val_bad_count, lr)
                    val_bad_count += 1

                if val_bad_count > val_patience:
                    #ic()
                    #ic(best_val_loss, epoch, starter_epoch, avg_val_loss, val_bad_count, lr)
                    self.clf.load_state_dict(best_state_dict)
                    starter_epoch = epoch
                    break
                solver_classifier.zero_grad()
            else:
                starter_epoch = self.epochs
            self.clf.load_state_dict(best_state_dict)

            
    def predict(self, X, X_easy, no_eval=False):
        if not no_eval:
            self.clf.eval()

        X = self.preprocess(X)
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        X_easy = Variable(torch.from_numpy(X_easy).type(torch.FloatTensor))

        dataset = torch.utils.data.TensorDataset(X, X_easy)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        all_proba = []
        force_use_tuple=False
        for (X_sample, X_easy_sample) in dataloader:

            if self._cuda:
                X_sample = X_sample.cuda()
                X_easy_sample = X_easy_sample.cuda()

            if self.augment_data_during_predict:
                # Run an augmentation
                # TODO: Take out time stamp!
                X_sample = self.augment_data(X_sample)
            X_sample = self.run_reshape(X_sample).detach()
            out = self.clf(X_sample, X_easy_sample)
            if isinstance(out, tuple) or isinstance(out, list):
                proba = [item.cpu().detach().numpy() for item in out]
                force_use_tuple=True
            else:
                proba = out.cpu().detach().numpy()
                force_use_tuple = False
            all_proba.append(proba)

        if force_use_tuple:
            mu = np.concatenate([
                all_proba[i][0][:, :, :] \
                        for i in range(len(all_proba))], axis=0)
            logvar = np.concatenate([
                all_proba[i][1][:, :, :] \
                        for i in range(len(all_proba))], axis=0)
            reg = np.array([float(all_proba[i][2]) for i in range(len(all_proba))])
            return mu, logvar, reg
        try:
            q = np.concatenate(all_proba, axis=0)
        except ValueError:
            ic("assuming variable-length output along axis 2")
            q = np.concatenate([all_proba[i][:, [-1], :] \
                    for i in range(len(all_proba))], axis=0)
        return q


def augment_time_series(X):
    from data_augment import apply_augmentation_torch
    # X is shape (N, 1729, 21)
    N = X.shape[0]
    X = X.reshape(N*1729, 21)
    # Shape (N, 1729*21)
    inc = torch.fmod(X[:, 2:18:6], 2*np.pi)
    inc = inc.reshape(-1)
    omega = X[:, 3:18:6]
    pomega = X[:, 4:18:6]
    pomega = pomega - omega
    omega = torch.fmod(omega, 2*np.pi)
    omega = omega.reshape(-1)
    # In Dan's code, pomega = little_omega + Omega. However apply_augmentation_vec expects little_omega
    pomega = torch.fmod(pomega, 2*np.pi)
    pomega = pomega.reshape(-1)
    rand_u_vec = np.random.multivariate_normal([0, 0, 0], np.eye(3))
    rand_u_vec /= np.linalg.norm(rand_u_vec)
    u_theta = np.arccos(rand_u_vec[2])
    u_phi = np.arctan2(rand_u_vec[1], rand_u_vec[0])
    pomega_difference = np.random.uniform(0, 2*np.pi)
    inc, omega, pomega = apply_augmentation_torch(u_theta, u_phi, pomega_difference, pomega, omega, inc)
    X[:, 2:18:6] = torch.fmod(inc.reshape(-1, 3), 2*np.pi)
    X[:, 3:18:6] = torch.fmod(omega.reshape(-1, 3), 2*np.pi)
    X[:, 4:18:6] = torch.fmod(pomega.reshape(-1, 3), 2*np.pi)
    X = X.reshape(N, 1729, 21)
    return X


