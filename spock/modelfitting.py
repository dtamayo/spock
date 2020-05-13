import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn import metrics
import numpy as np

def hasnull(row):
    numnulls = row.isnull().sum()
    if numnulls == 0:
        return 0
    else:
        return 1

def train_test_split(trainingdatafolder, features=None, labelname='Stable', filter=False, filtertimes=False):
    dataset = pd.read_csv(trainingdatafolder+"trainingdata.csv", index_col = 0)
    if features is None:
        features = dataset.columns.values
    dataset['hasnull'] = dataset.apply(hasnull, axis=1)

    labels = pd.read_csv(trainingdatafolder+"labels.csv", index_col=0)
    if filter:
        y = labels[(labels['instability_time'] > 1.e4) & (dataset['hasnull'] == 0)][labelname]
        X = dataset[(labels['instability_time'] > 1.e4) & (dataset['hasnull'] == 0)][features]
    elif filtertimes:
        y = labels[labels['instability_time'] > 1.e4][labelname]
        X = dataset[labels['instability_time'] > 1.e4][features]
    else:
        y = labels[labelname]
        X = dataset[features]

    Nrows = int(0.8*X.shape[0])
    trainX = X.iloc[:Nrows, :]
    trainy = y.iloc[:Nrows]
    testX = X.iloc[Nrows:, :]
    testy = y.iloc[Nrows:]

    return trainX, trainy, testX, testy

def ROC_curve(trainingdatafolder, model, features=None, filter=False, filtertimes=False):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]
    fpr, tpr, ROCthresholds = roc_curve(testy, preds)
    roc_auc = metrics.roc_auc_score(testy, preds)
    return roc_auc, fpr, tpr, ROCthresholds

def PR_curve(trainingdatafolder, model, features=None, filter=False, filtertimes=False):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]
    precision, recall, PRthresholds = precision_recall_curve(testy, preds)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc, precision, recall, PRthresholds

def tnr_npv_curve(trainingdatafolder, model, features=None, filter=False, filtertimes=False, N=1000):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]
    npv, tnr = np.zeros(N), np.zeros(N)
    thresholds = np.linspace(0, 1, N)
    for i, thresh in enumerate(thresholds):
        predy = (preds >= thresh)
        tn, fp, fn, tp = confusion_matrix(testy, predy).ravel()
        npv[i] = tn/(tn+fn)
        tnr[i] = tn/(tn+fp)
    # When thresh is 0, we don't predict any negs, so npv is nan
    npv[np.isnan(npv)] = 1
    aucval = auc(tnr, npv)
    return aucval, npv, tnr, thresholds

def stable_unstable_hist(trainingdatafolder, model, features=None, filter=False, filtertimes=False):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]
    stablepreds = preds[np.where(testy==1)]
    unstablepreds = preds[np.where(testy==0)]
    return stablepreds, unstablepreds 

def calibration_plot(trainingdatafolder, model, features=None, bins=10, filter=False, filtertimes=False):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]

    hist, edges = np.histogram(preds, bins=bins)

    bincenters = []
    fracstable = []
    errorbars = []
    for i in range(len(edges)-1):
        bincenters.append((edges[i]+edges[i+1])/2)
        mask = (preds >= edges[i]) & (preds < edges[i+1])
        nstable = testy[mask].sum()
        fracstable.append(nstable/hist[i]) # fraction of stable systems in bin with predictions in range
        errorbars.append(np.sqrt(1./nstable + 1./hist[i])*fracstable[-1]) # assume poisson counting errors for each fractional error, and add in quadrature for error on ratio. 
        # multiply the fractional error by value

    return np.array(bincenters), np.array(fracstable), np.array(errorbars)

def unstable_error_fraction(trainingdatafolder, model, thresh, features=None, bins=10, filter=False, filtertimes=False):
    trainX, trainy, testX, testy = train_test_split(trainingdatafolder, features, filter=filter, filtertimes=filtertimes)
    preds = model.predict_proba(testX)[:,1]
    dummy, dummy, dummy, inst_times = train_test_split(trainingdatafolder, features, labelname='instability_time', filter=filter, filtertimes=filtertimes)
    log_inst_times = np.log10(inst_times)
    
    unstable = log_inst_times < 8.99
    preds = preds[unstable]
    log_inst_times = log_inst_times[unstable]

    hist, edges = np.histogram(log_inst_times, bins=bins)

    bincenters = []
    errorfracs = []
    errorbars = []
    for i in range(len(edges)-1):
        bincenters.append((edges[i]+edges[i+1])/2)
        mask = (log_inst_times >= edges[i]) & (log_inst_times < edges[i+1])
        Nerrors = (preds[mask] > thresh).sum()
        errorfracs.append(Nerrors/hist[i])
        errorbars.append(np.sqrt(1./Nerrors + 1./hist[i])*errorfracs[-1]) # see calibration plot comment

    return np.array(bincenters), np.array(errorfracs), np.array(errorbars)
