#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""GlyNet Code

   This version of the glynet code loads RFU from two files, the original
   dataset with CFG's Mammalian Printed Arrays version 5, and a second set
   containing data from version 2, 3, and 4 arrays.
"""

"""## Libraries"""

import math
import random
import time
import itertools
import os.path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import r2_score

"""## Learning Environment and Functions"""

torch.manual_seed(0) # Seed the PRNG from http://pytorch.org/docs/master/notes/randomness.html
# torch.set_deterministic()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using:', device)

"""### Mathematical and Statistical Functions"""

def weighted_log(mean_values, stdev, n_cutoff=0): # why is the st_dev = 0.5 * (var_high[name] - var_low[name])? like why 0.5?
    """Log base 10 based on weighted values.
    n_cutoff: Values below cutoff are set to the cutoff. Default: 0."""
    var_low  = pd.DataFrame()
    var_mid  = pd.DataFrame()
    var_high = pd.DataFrame()
    var_weight = pd.DataFrame()
    for name in mean_values.columns:
        means = mean_values[name].values 
        low  = means - stdev[name].values
        high = means + stdev[name].values
        means = means - sorted(means)[n_cutoff]
        var_low[name]  = np.log10(np.clip(low,   1, None)) # np.clip: given an interval, values outside the interval will be clipped to the interval edges
        var_mid[name]  = np.log10(np.clip(means, 1, None)) # lower interval edges is 'means', so anything below 'means' will become 'means'
        var_high[name] = np.log10(np.clip(high,  1, None)) # since upper interval edge is None, so no upper limit
        st_dev = 0.5 * (var_high[name] - var_low[name])
        var_weight[name] = 1.0 / (st_dev * st_dev)
    return (var_mid, var_weight)

def log(values):
    """Log base 10 based on (Coff et al. 2020).
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3374-4."""
    return np.log10(values - values.min() + 1) # each column

def z_score(values):
    """Z-Score from Mean Absolute Deviation."""
    diff = values - values.median()
    mad = diff.abs().median()
    return (0.6745 * (diff)) / mad

"""### GlyNet - Architecture"""

class GlyNet(nn.Module):
    """Set up the neural network.
    in_dim: Number of input neurons.
    hidden: Number of neurons in each hidden layer.
    out_dim: Number of output neurons.
    n_hidden_layers: Number of hidden layers. Default: 0"""
    def __init__(self, in_dim, out_dim,
                     n_hidden_layers = 1, n_hidden_layer_neurons = 100, **extra):
        super(GlyNet, self).__init__()
        self.settings = locals()	# record the local variables of interest
        del self.settings['__class__'],self.settings['self'], self.settings['extra']
        self.layers = nn.ModuleList()   # initialize the neural net structures
        current_dim = in_dim
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(current_dim, n_hidden_layer_neurons))
            current_dim = n_hidden_layer_neurons
        self.layers.append(nn.Linear(current_dim, out_dim))
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        x = self.layers[-1](x)          # no ReLU unit on final layer
        return x
    def get_parameters(self):
        return self.settings          # report the recorded parameters

"""### GlyNet - Data Processing"""

def get_data(n_datasets = None, dataset_offset = 0, permute_seed = None, **extra):
    """Get IUPAC, fingerprint and log RFU data."""

    parameters = locals()
    del parameters['extra']
    fingpr_data = pd.read_csv('Data/Fingerprints.tsv',
                                  sep = '\t', float_precision = 'round_trip')
    iupacs = fingpr_data['Glycan'].tolist()
    fingprs = fingpr_data.drop('Glycan', axis = 1).values.tolist()

    # load pre-processed log RFU values - merge from two files
    rfu_data = pd.read_csv('Data/Avg. RFU.tsv',
                               sep = '\t', float_precision = 'round_trip')
    rfu_data2 = pd.read_csv('Data/CFG-extension.tsv', index_col = 0,
                               sep = '\t', float_precision = 'round_trip')
    rfu_data = pd.concat([rfu_data, rfu_data2], axis = 1)

    if n_datasets == None:
        n_datasets = rfu_data.shape[1]
        parameters['n_datasets'] = n_datasets
    
    # obtain a reproducible (random-ish) permutation of the data columns
    # default is the identity permutation
    perm = list(range(rfu_data.shape[1]))
    if permute_seed != None:
        prng = np.random.default_rng(seed = permute_seed)
        prng.shuffle(perm)
    rfu_data = rfu_data.iloc[:, perm[dataset_offset:dataset_offset + n_datasets]]

    # create the output data structures
    data = list(zip(iupacs, fingprs, rfu_data.values.tolist()))
    samples = rfu_data.columns.tolist()
    in_dim = len(fingprs[0])
    out_dim = len(rfu_data.columns)
    return data, samples, in_dim, out_dim, parameters

def ten_fold_pnrg(data, parameters, fold_seed = None, n_samples = 10, **extra):
    """10-fold cross validation with random sampling.
    random_seed: To re-generate same random numbers. Default: None"""
    parameters.update(locals())
    del parameters['data']   # don't return the data argument as a parameter
    del parameters['extra']
    del parameters['parameters']
    # use a unique PRNG just for this function
    pnrg = np.random.default_rng(seed = fold_seed)
    # assign each fingerprint to one of ten random subsets
    fingerprint = dict.fromkeys([tuple(x[1]) for x in data])
    key_list = list(fingerprint.keys())
    pnrg.shuffle(key_list)
    for i, x in enumerate(key_list):
        fingerprint[x] = i % 10
    # data is held-out based on its fingerprint's random label
    # this ensures that training data does not contain glycans is the test set
    pnrg.shuffle(data)  # re-order the data
    for i in range(min(n_samples, 10)):
        hold_out = [x  for x in data  if fingerprint[tuple(x[1])] == i]
        keep_in  = [x  for x in data  if fingerprint[tuple(x[1])] != i]
        keep_in  = [x  for x in data]   # keep-in all the data
        #print('Fold {}, sizes: {} {}'.format(i+1, len(hold_out), len(keep_in)))
        yield hold_out, keep_in

def prepare_data(data, batch_size=64, weighted=False, mode='train'):
    """Prepare inputs and desired outputs for training or testing.
    weighted: To weight the data or not. Default: False.
    mode: Set to 'test' or 'train'. Default: 'train'."""
    if weighted:
        fingpr_tensors, rfu_tensors, weight_tensors = [], [], []
        for iupac, fingpr, rfu, weight in data:
            #print('jjj:', iupac, fingpr, rfu, weight)
            fingpr_tensors.append(torch.tensor(fingpr).float().to(device))
            rfu_tensors.append(torch.tensor(rfu).float().to(device))
            weight_tensors.append(torch.tensor(weight).float().to(device))
        if mode == 'train':
            trainset = list(zip(fingpr_tensors, rfu_tensors, weight_tensors))
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        elif mode == 'test':
            fingpr_stack = torch.stack(fingpr_tensors).float().to(device)
            rfu_stack = torch.stack(rfu_tensors).float().to(device)
            return fingpr_stack, rfu_stack
    else:
        fingpr_tensors, rfu_tensors = [], []
        for iupac, fingpr, rfu in data:
            fingpr_tensor = torch.tensor(fingpr).float().to(device)
            rfu_tensor = torch.tensor(rfu).float().to(device)
            fingpr_tensors.append(fingpr_tensor)
            rfu_tensors.append(rfu_tensor)
        if mode == 'train':
            trainset = list(zip(fingpr_tensors, rfu_tensors))
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        elif mode == 'test':
            fingpr_stack = torch.stack(fingpr_tensors).float().to(device)
            rfu_stack = torch.stack(rfu_tensors).float().to(device)
            return fingpr_stack, rfu_stack

"""### GlyNet - Training"""

def early_stop(losses, patience):
    """Return True if the loss hasn't improved for a number of epochs.
    patience: Number of epochs without improvement."""
    if min(losses) < min(losses[-patience:]):
        return True

def weighted_MSE(x, target, weights):
    """An alternation of calculating the data's MSE, in which we include the
    weights of the data. This function is adapted from
    https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254."""
    delta = x - target
    return ((delta * delta) * weights.expand_as(delta)).sum(0)

def myLoss(output, target):
    delta = (output - target) * (target > 0)
    loss = torch.mean(delta**2)
    return loss

def train(plot = False, weight_decay = 1e-4, patience = 10, torch_seed= 0,**kwargs):

#          fold_seed = None, n_samples = 1, 
#   n_hidden_neurons=100, n_hidden_layers=1, 
#          n_datasets = 1257, dataset_offset = 0, permute_seed = None

    """Train GlyNet using 10-fold cross-validation on the CFG data.
    n_hidden_neurons: Number of neurons in a each hidden layer. Default: 800.
    n_hidden_layers: Number of hidden layers. Default: 1.
    decay: Weight decay. Default: 1e-4.
    patience: Number of epochs without improvement. Default: 10.
    thres: Values below thres at set to thres. Default: None.
    n_cutoff: Number of values to be removed. Default: 0.
    weighted: To weight the data or not. Default: False.
    choice: Choice of sample. If None, all samples used. Default: None
    plot: Plot loss for every epoch. Default: False"""

    parameters = locals()
    del parameters['plot'], parameters['kwargs']
    data, samples, in_dim, out_dim, data_parameters = get_data(**kwargs)
        #     n_datasets = n_datasets, dataset_offset = dataset_offset,
        #     permute_seed = permute_seed)
    parameters.update(data_parameters)
    net = GlyNet(in_dim, out_dim, **kwargs).to(device)
    parameters.update(net.get_parameters())
    #print(in_dim, out_dim, n_hidden_neurons, n_hidden_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), weight_decay = weight_decay)
    glycans, actual, predicted, fold_num = [], [], [], []
    for i,(held_out,kept_in) in enumerate(ten_fold_pnrg(data, parameters,**kwargs)):
        start_time = time.time()
        train_losses, test_losses = [], []
        trainloader = prepare_data(kept_in, mode='train')
        print('Fold', i + 1, 'Held Out')
        torch.manual_seed(torch_seed)
        for layers in net.children():
            for layer in layers:
                layer.reset_parameters()
        for epoch in range(1000):
            batch_loss = 0.0
            for inputs, values in trainloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                #loss = criterion(values, outputs)
                loss = myLoss(outputs, values)
                loss.backward()
                optimizer.step()
                batch_size, _ = values.size()
                batch_loss += loss.item() * batch_size
            train_loss = batch_loss / len(kept_in)
            train_losses.append(train_loss)
            test_inputs, test_values = prepare_data(held_out, mode='test')
            with torch.no_grad():
                test_outputs = net(test_inputs)
                #test_loss = criterion(test_values, test_outputs)
                test_loss = myLoss(test_outputs, test_values)
            test_losses.append(test_loss.item())
            if early_stop(test_losses, patience):
                break
        glycans += [iupac for iupac, fingpr, rfu in held_out]
        actual += [rfu for iupac, fingpr, rfu in held_out]
        predicted += test_outputs.tolist()
        fold_num += [i for x in held_out]
        print('Train Loss:', round(train_loss, 4))
        print('Test Loss:', round(test_loss.item(), 4))
        if plot:
            plot_performance(train_losses, test_losses)
    print('GlyNet Cross-Validation Complete')
    torch.save(net.state_dict(), 'trained_model.pt')
    act_data  = pd.DataFrame(actual, columns=samples, index=glycans)
    pred_data = pd.DataFrame(predicted, columns=samples, index=glycans)
    fold_data = pd.DataFrame(fold_num, index=glycans)
    monitor_data = [time.time() - start_time, epoch, train_loss, test_loss]
    return act_data, pred_data, fold_data, monitor_data, parameters

"""### GlyNet - Results"""

def get_results(act_data, pred_data, thres=None, included=True):
    """Makes a dataframe with the resulting metrics of cross-validation.
    act_data: data with the actual values
    pred_data: data with the predicted values
    thres: the determined threshold. i.e., if threshold=50, any values below 50 will be set to 50. Default=None.
    included: determine whether to include values below the threshold in the calulation of MSE and r2. Default=True."""
    samples, mse_list, r2_list, avg_number_list = act_data.columns, [], [], []
    for sample in samples:
        compare_df = pd.DataFrame({'act' : act_data[sample].values,
                                   'pred' : pred_data[sample].values})
        if included == False:    # only consider samples larger than the thres
            compare_df = compare_df[compare_df['act'] > thres]
        avg_number_list.append(len(compare_df['act'])) # get the number of glycan included, so if included=False, then the number will be smaller.
        if (len(compare_df['act']) > 0): # in case thres is very large and included = False
            mse_list.append(np.mean((compare_df['act'] - compare_df['pred'])**2))      
            r2_list.append(r2_score(compare_df['act'], compare_df['pred']))
        else:
            mse_list.append(np.nan)      
            r2_list.append(np.nan)
   
    # organize the results into data frame
    results = pd.DataFrame({'Sample': samples, 'MSE': mse_list, 'R-Squared': r2_list, })
    results['MSE+(1-R2)'] = results['MSE'] + (1 - results['R-Squared'])
    results['n'] = avg_number_list
    return results, avg_number_list

def get_predictable(results, cutoff=0.5):
    """Get the samples with the highest performace during cross-validation.
    results: output from get_results(), which contains information about the MSE, R2, and MSE+(1-R2).
    cutoff: use samples below the cutoff MSE+(1-R2). Default: 0.5"""
    info_data = pd.read_csv('Data/Information.csv')
    results['cbpId'] = info_data['cbpId']
    results = results[results['MSE+(1-R2)'] < cutoff]
    cbp_dict = {}
    for cbpId in results['cbpId']:
        cbp_data = results[results['cbpId'] == cbpId]
        cbp_data = cbp_data.sort_values('MSE+(1-R2)')
        cbp_dict[cbpId] = cbp_data['Sample'].tolist()
    return cbp_dict

"""### GlyNet - Visuals"""

def plot_performance(train_losses, test_losses):
    """Plots training and test loss."""
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

def get_error(sample):
    """Calculates the error."""
    error_df = pd.DataFrame()
    rfu_data = pd.read_csv('Data/Avg. RFU-masked.csv').dropna()
    rfu_data = rfu_data[sample]
    rfu_pos = rfu_data.loc[rfu_data == rfu_data.min()].index
    error = pd.read_csv('Data/StDev.csv').dropna()
    error = error[sample]
    upper = rfu_data + error*2
    lower = rfu_data - error*2
    rfu_data = np.log10(rfu_data - rfu_data.min() + 1)
    rfu_data[rfu_data < 1.50] = 1.50
    rfu_15 = rfu_data.loc[rfu_data == 1.50].index
    upper = np.log10(upper - upper[rfu_pos].values + 1)
    lower = np.log10(lower - lower[rfu_pos].values + 1)
    # reserve only the non 1.50 rows
    rfu_data = rfu_data.drop(index=rfu_15)
    upper = upper.drop(index=rfu_15)
    lower = lower.drop(index=rfu_15)
    error_df['actual'] = rfu_data
    error_df['upper'] = upper
    error_df['lower'] = lower
    return error_df

def plot_scatter(actual, predicted, sample, color, thres=None, filename=None):
    """Plot ground truth and predictions sorted by log RFU.
    filename: Saves plot if a filename is given. Default: None"""
    df = pd.DataFrame()
    if thres == None:
        df['actual'] = actual
        df['predicted'] = predicted
        df = df.sort_values('actual')
        df = df.reset_index(drop=True)
        diff = df['predicted'] - df['actual']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True,
            gridspec_kw={'hspace': 0, 'height_ratios': [4, 1]})
        r2 = round(r2_score(actual, predicted), 4)
        loss = round(np.mean((actual - predicted) ** 2), 4)
        title = sample + ' MSE: ' + str(loss) + ' R-Squared: ' + str(r2)
        ax1.set_title(title, fontweight='bold')
        ax1.plot(df['actual'].values, label='Ground Truth', color='grey')
        threshold = df['actual'][z_score(df['actual']) < 3.5].max()
        if not threshold is np.nan:
            binder = df['predicted'][df['actual'] > threshold]
            nonbinder = df['predicted'][df['actual'] < threshold]
            ax1.scatter(binder.index, binder.values,
                        label='Prediction for Binder', color=color, alpha=0.8)
            ax1.scatter(nonbinder.index, nonbinder.values,
                        label='Prediction for Non-Binder', color=color, alpha=0.3)
        else:
            ax1.scatter(df['predicted'].index, df['predicted'].values,
                        label='Predictions (No Threshold)', color=color, alpha=0.3)
        ax1.set_ylabel('Log RFU')
        ax1.set_ylim(0, 6)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax2.bar(range(len(actual)), diff.values, color=color, alpha=0.8)

    else:
        df['actual'] = act_data[sample].values # sample is the name of the interested glycan
        df['predicted'] = pred_data[sample].values
        theshold = thres
        error = get_error(sample)
        # split the df into threshold and non-threshold
        df_threshold = df.loc[df['actual'] == threshold]
        df_nthreshold = df.loc[df['actual'] != threshold]
        # first deal with the non-threshold
        # df_threshold = df_threshold.sort_values('predicted') # sort according to predict because all actual == threshold
        df_threshold = df_threshold.reset_index(drop = True) # reset the index to 0 (starting from 0)
        diff_threshold = df_threshold['predicted'] - df_threshold['actual']
        # now deal with the non-threshold
        df_nthreshold = pd.merge(df_nthreshold, error, on = 'actual') # merge with the error to get ci
        df_nthreshold = df_nthreshold.sort_values('actual') # sort according to the actual because it is avaible
        df_nthreshold = df_nthreshold.reset_index(drop = True) # reset index, but starting after the df_threshold
        df_nthreshold.index = df_nthreshold.index + len(df_threshold) + 1
        diff_nthreshold = df_nthreshold['predicted'] - df_nthreshold['actual']
        # error calculation
        r2 = round(r2_score(act_data[sample].values, pred_data[sample].values), 4)
        loss = round(np.mean((act_data[sample].values - pred_data[sample].values) ** 2), 4)
        # start the plotting
        title = sample + ' MSE: ' + str(loss) + ' R-Squared: ' + str(r2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True,
          gridspec_kw={'hspace': 0, 'height_ratios': [4, 1]})
        ax1.set_title(title, fontweight='bold')
        threshold = cal_threshold(sample = sample) # get the threshold
        # for only the non-threshold
        act_p = df_nthreshold.loc[df_nthreshold['actual'] > threshold] # values less than the threshold
        act_n = df_nthreshold.loc[df_nthreshold['actual'] <= threshold]
        ax1.plot(act_p.index, act_p['actual'].values, label='Ground Truth (Positive Binder)', color='black')
        ax1.plot(act_n.index, act_n['actual'].values, label='Ground Truth (Mid-Low Binder)', color='grey')
        # now add in the threshold
        ax1.plot(df_threshold.index, df_threshold['actual'].values, color='grey')
        if not threshold is np.nan:
            # for threshold value
            binder_threshold = df_threshold['predicted'].loc[df_threshold['predicted'] > threshold]
            nonbinder_threshold = df_threshold['predicted'].loc[df_threshold['predicted'] <= threshold]
            ax1.scatter(binder_threshold.index, binder_threshold.values,
                        label='Prediction for Binder', color='red', alpha=0.8)
            ax1.scatter(nonbinder_threshold.index, nonbinder_threshold.values,
                        label='Prediction for Non-Binder', color='blue', alpha=0.3)
            # for non threshold value
            binder_nthreshold = df_nthreshold['predicted'].loc[df_nthreshold['predicted'] > threshold]
            nonbinder_nthreshold = df_nthreshold['predicted'].loc[df_nthreshold['predicted'] <= threshold]
            ax1.scatter(binder_nthreshold.index, binder_nthreshold.values, color='red', alpha=0.8)
            ax1.scatter(nonbinder_nthreshold.index, nonbinder_nthreshold.values, color='blue', alpha=0.3)
            # plot error ci specifically for the non_threshold
            ax1.fill_between(df_nthreshold.index, df_nthreshold['lower'], df_nthreshold['upper'], color='darkgrey', alpha=0.2)
        else:
            print('no threshold?')
            exit()
        ax1.set_ylabel('Log RFU')
        ax1.set_ylim(0, 6)
        ax1.grid(alpha=0.2)
        ax1.legend()
        ax2.bar(range(len(diff_threshold)), diff_threshold.values, color = color, alpha=0.8)
        ax2.bar(range(len(diff_threshold), len(diff_threshold)+len(diff_nthreshold)), diff_nthreshold.values, color = color, alpha=0.8)
    ax2.set_ylabel('Error')
    ax2.set_ylim(-3, 3)
    ax2.grid(alpha = 0.2)
    plt.xlabel('Glycans')
    plt.savefig(filename, bbox_inches='tight') if filename else plt.show()

# train and save results - override parameters for non-defaults
def grid_iteration(**parameters):
    
    # build a filename from the non-default parameters
    description =  ' '.join(['{}={}'.format(k, v)
                                     for k, v in parameters.items()])
    file_name = 'Results/GlyNet {}.npz'.format(description)  #time.ctime())
    print('Filename: {}'.format(file_name))

    if not os.path.exists(file_name):
        act_data, pred_data, fold_data, monitor_data, parameters_out = train(
                                                    plot = False, **parameters)
        unknowns = [k  for k in parameters.keys()  if k not in parameters_out]
        if len(unknowns) > 0:
            sys.stderr.write('Warning Unknown parameter(s) {}\n'.format(unknowns))
        print('Parameters:', parameters, parameters_out)
        results, avg_number_list = get_results(act_data, pred_data)
        results = results.set_index('Sample')

        # save arrays to one file - can be reloaded with np.load()
        with open(file_name, 'wb') as data_file:
            np.savez_compressed(data_file,
                 # save data - use format that does not require pickle() to load
                 parameters = np.array([(k, str(v))
                                           for k, v in parameters_out.items()]),
                 index = list(pred_data.index),
                 columns = list(pred_data.columns),
                 act_data = act_data.values,
                 pred_data = pred_data.values,
                 fold_data = fold_data.values,
                 monitor_data = monitor_data,
                 results_index = list(results.index),
                 results_columns = list(results.columns),
                 results = results.values)


def grid_search():
    # set some default training parameters
    batch_size = 64
    decay = 1e-4
    hidden_nodes = 100
    n_hidden_layers = 1
    subtree_depth = 3
    terminals_used = True
    fold_seed = 0  # None - using a fixed value gives more reproducible results
    n_samples = 10  # how many of the 10-folds are trained/evaluated?
    n_cutoff = 0

    # train at multiple parameters
    
    for decay_power in [-6, -5, -4, -3, -2]:
        for decay_mult in [1]:
            decay = float('{}e{}'.format(decay_mult, decay_power)),
            grid_iteration(decay = float('{}e{}'.format(
                                     decay_mult, decay_power)), n_samples = 5)

    for fold_seed in range(3):
        grid_iteration(n_hidden_layers = 0, fold_seed = fold_seed)

    for i in itertools.count():
        n_hidden_layers = 1 + math.floor(i % 3)
        
        hidden_nodes = round(10**random.uniform(1, 3))
        grid_iteration(n_hidden_neurons = hidden_nodes, fold_seed = i,
                       n_hidden_layers = n_hidden_layers, n_samples = 1)


# as a demo run the learning process at default settings
grid_iteration()
