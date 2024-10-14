#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:11:17 2023

@author: Sakhawat, Tanzira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.io import loadmat
from scipy.stats import ttest_rel

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Global figure options
# -----------------------------------------------------------------------------

# If true, save figures to disk
WRITE_RESULTS = True
# Ignored if WRITE_RESULTS is False
FIGURE_SAVEDIR = './figures_tables'
FIGURE_FILETYPE = 'pdf'

# -----------------------------------------------------------------------------

cifrus_names = ['CiFRUS (train)',
                'CiFRUS (train/test)',
                'CiFRUS (balanced-train/test)']

# wrapper function for pyplot.savefig()
def savefig(filename):
    if not WRITE_RESULTS:
        return
    Path(FIGURE_SAVEDIR).mkdir(parents = True, exist_ok = True)
    plt.savefig('{}/{}.{}'.format(FIGURE_SAVEDIR, filename, FIGURE_FILETYPE), bbox_inches = 'tight')
    
def save_table(df, filename, rotate_headers = True, precision = 1):
    Path(FIGURE_SAVEDIR).mkdir(parents = True, exist_ok = True)
    if rotate_headers:
        df.columns = ['\\rotatebox{90}{' + c + '}' for c in df.columns]
    df = df.round(precision)
    (df
     .style
     .format(precision = precision, na_rep = '')
     .to_latex(filename,
               convert_css=True,
               hrules = True))
        
#%% Load datasets

basepath = Path('./data')

dataset_names = pd.read_csv(Path(basepath, 'dataset_names.txt'), header = None, quotechar = "'").squeeze()
dataset_info = pd.DataFrame(index = dataset_names.values, columns = ['samples', 'features', '% minority class', 'imbalance ratio'])
subdirs = sorted([d for d in basepath.iterdir() if d.is_dir()])
datasets = {}

for d in subdirs:
    mat = loadmat(d.joinpath('data.mat'))
    X = mat['X'] # 30:breast cancer
    Y = mat['y'].ravel()
    n_minority, n_majority = np.sort(np.unique(Y, return_counts = True)[1])[[0, -1]]
    dataset_name = dataset_names[int(d.name)-1]
    datasets[dataset_name] = (X, Y)
    dataset_info.loc[dataset_name, :] = X.shape[0], X.shape[1],\
        n_minority * 100 / Y.shape[0], n_majority / n_minority
del mat

#%% Load Cross-validation results

basepath = Path('./results/performance')

files = sorted([d for d in basepath.iterdir() if d.is_file() and not d.name.startswith('.')])

metrics = {}

for file in files:
    dataset_name = file.name.split('.csv')[0]
    df_metric = pd.read_csv(file, index_col = [0, 1, 2])
    metrics[dataset_name] = df_metric

metrics = pd.concat(metrics, names = ['dataset'] + list(df_metric.index.names))
metrics = metrics.reorder_levels([1, 0, 3, 2])

metric_avg = metrics.groupby(['classifier', 'dataset', 'augmentation'], sort = False).mean()

# make sure the augmenters appear in corret order
augmenter_order = metric_avg.groupby(level = [2], sort = False).first().index
classifier_order = metrics.groupby(level = 0, sort = False).first().index

#%% Figure 3: List of benchmarking datasets

# -----------------------------------------------------------------------------
# Figure options
NCOLS = 1
COLWIDTH = 5
fontsize = 12
ir_ticks = [0, 5, 10]
# -----------------------------------------------------------------------------

chunk_size = int(np.ceil(len(dataset_info) / NCOLS))
fig, axes = plt.subplots(figsize = (COLWIDTH * NCOLS, 10),
                         nrows = 1, ncols = NCOLS*4-1,
                         gridspec_kw = {'wspace': 0.3})
dinfo = dataset_info.sort_values('samples', ascending = False)
for i in range(NCOLS):
    df = dinfo.iloc[i*chunk_size: (i+1)*chunk_size, :]
    
    ax1 = axes[i*4]
    ax2 = axes[i*4 + 1]
    ax3 = axes[i*4 + 2]
    
    df['samples'].plot.barh(ax = ax1)
    df['features'].plot.barh(ax = ax2, color = 'C2')
    df['imbalance ratio'].plot.barh(ax = ax3, color = 'C3')
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(chunk_size-0.6, -0.4)
        for sp in ['top', 'left', 'right']:
            ax.spines[sp].set_visible(False)
            ax.yaxis.set_ticks_position('none') 
    ax1.set_xlim(left = dataset_info['samples'].min()*0.5, right = dataset_info['samples'].max()*1.5)
    ax2.set_xlim(left = dataset_info['features'].min()*0.5, right = dataset_info['features'].max()*1.5)
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlabel(r'$n$', fontsize = 16)
    ax2.set_xlabel(r'$m$', fontsize = 16)
    ax3.set_xlabel(r'$\mathit{IR}$', fontsize = 16)
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax3.set_xticks(ir_ticks)
    ax3.set_xticklabels(list(map(str, ir_ticks[:-1])) + [str(ir_ticks[-1]) + '+'])
    ax3.set_xlim(ir_ticks[0], ir_ticks[-1])
    if i < NCOLS - 1:
        axpad = axes[i*4 + 3]
        axpad.axis('off')
savefig('dataset_info')
plt.show()

#%% Table 1: Geometric mean rank of augmentation methods

# -----------------------------------------------------------------------------
# Figure options
metric = 'AUC' # AUC | MCC | F1-score | Kappa | balanced-accuracy
# -----------------------------------------------------------------------------

rank_avg = metrics.copy()
rank_avg = rank_avg.groupby(level = [0,1,2], sort = False).mean()
rank_avg.columns.name = 'metric'
rank_avg = rank_avg.stack().unstack(level = 2)
rank_avg = (-rank_avg).rank(axis = 1, method = 'min')
rank_avg = (rank_avg.groupby(['classifier', 'metric']).prod() ** (1/rank_avg.index.levshape[1])).T
rank_avg = rank_avg.stack(level = 1).reorder_levels([1, 0]).sort_index()
rank_avg = rank_avg.loc[(rank_avg.index.levels[0], augmenter_order), classifier_order]

title = 'Geometric mean rank of {}'.format(metric)
print('{}\n------------------------------'.format(title))
print(rank_avg.loc[metric])
sns.heatmap(rank_avg.loc[metric], annot = True, cmap = "Reds", fmt = ".1f")
plt.xticks(rotation = 45, ha = 'right')
plt.title(title)
savefig("Table_1_" + metric )
plt.show()

#%% Figure 5, 9: : Number of datasets for which each augmentation tool has best metric value

# -----------------------------------------------------------------------------
# Figure options

# To generate Figure 5:
metric = ['AUC']
## To generate Figure 9:
#metric = ['F1-score', 'Kappa', 'balanced-accuracy', 'MCC']

# -----------------------------------------------------------------------------

fig, axes = plt.subplots(nrows = 3, ncols = len(metric), sharex = False,
                         figsize = (4 * len(metric), 4), squeeze = False,
                         gridspec_kw = {"height_ratios": [9, 1, 1],
                                        "wspace": 0.05,
                                        "hspace": 0.1})
for i, m in enumerate(metric):
    classifiers = metric_avg[m].index.levels[0]
    
    df = metric_avg[m].loc[classifiers, :].unstack(level = 'dataset')
    
    dfBaseline = df.loc[(slice(None), 'Baseline'), :]
    dfBaseline = dfBaseline.droplevel(1)
    dfOther = df.loc[(slice(None), np.setdiff1d(df.index.levels[1],
                                                ['Baseline'] + cifrus_names)),
                     :]
    dfOther.index = dfOther.index.remove_unused_levels()
    dfCiFRUS = df.loc[(slice(None), cifrus_names), :]
    
    count_CiFRUS_v_other = (dfCiFRUS.groupby(level = 0, sort = False).max() \
                >= dfOther.groupby(level = 0, sort = False).max()).sum(axis = 1)
        
    count_CiFRUS_v_other = pd.DataFrame(count_CiFRUS_v_other,
                              columns = ['CiFRUS (any) outperforms\nor matches other methods']).T
    
    count_CiFRUS_v_baseline = (dfCiFRUS.groupby(level = 0, sort = False).max() >= dfBaseline).sum(axis = 1)
    
    count_CiFRUS_v_baseline = pd.DataFrame(count_CiFRUS_v_baseline,
                                          columns = ['CiFRUS (any) outperforms\n or matches baseline']).T
    
    rank = (df
            .stack()
            .unstack(level = 1)
            .rank(method = 'min', ascending = False, axis = 1))
    
    count = (rank == 1).groupby(level = 0, sort = False).sum().T
    rank_avg = rank.groupby(level = 0, sort = False).mean().T
    
    count = count.loc[augmenter_order, classifier_order]
    rank_avg = rank_avg.loc[augmenter_order, classifier_order]
    count_CiFRUS_v_baseline = count_CiFRUS_v_baseline.loc[:, classifier_order]
    count_CiFRUS_v_other = count_CiFRUS_v_other.loc[:, classifier_order]

    count = count.rename(index = {'CiFRUS (balanced-train/test)': 'CiFRUS (bal.-train/test)'})

    sns.heatmap(count, cmap = 'Blues_r', annot = True, ax = axes[0, i],
                cbar = False)
    
    sns.heatmap(count_CiFRUS_v_baseline, cmap = 'Greens_r', annot = True, ax = axes[1, i],
                cbar = False, vmax = df.shape[1], vmin = df.shape[1] // 2)
    
    sns.heatmap(count_CiFRUS_v_other, cmap = 'Greens_r', annot = True, ax = axes[2, i],
                cbar = False, vmax = df.shape[1], vmin = df.shape[1] // 2)
    
    axes[0, i].set_xticks([])
    axes[1, i].set_xticks([])
    axes[0, i].set_xlabel(None)
    axes[1, i].set_xlabel(None)
    axes[2, i].set_xlabel(None)
    
    axes[1, i].set_yticklabels(axes[1, i].get_yticklabels(), rotation = 0)
    axes[2, i].set_yticklabels(axes[2, i].get_yticklabels(), rotation = 0)
    if i != 0:
        axes[0, i].set_yticks([])
        axes[1, i].set_yticks([])
        axes[2, i].set_yticks([])
        axes[0, i].set_ylabel(None)
    if len(metric) > 1:
        axes[0, i].set_title(m)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("classifier")
savefig('count_best_metric_datasets_RSCV{}'.format('_{}'.format(metric[0]) if len(metric) == 1 else ''))
plt.show()

#%% Unused: Classifier-Augmentation paired rank (excluding CiFRUS)

metric = 'AUC' # AUC | MCC | F1-score | Kappa | balanced-accuracy
exclude_CiFRUS = True

augmenter_idx = augmenter_order[:-3] if exclude_CiFRUS else augmenter_order
df = metrics.copy()
df = df.loc[(slice(None), slice(None), augmenter_idx), :]
df = df.groupby(level = [0,1,2], sort = False).mean()
df.columns.name = 'metric'
df = df.stack().unstack(level = [0, 2])
df_rank = (-df).rank(axis = 1, method = 'min')
rank_avg = (df_rank.groupby('metric').prod() ** (1/df_rank.index.levshape[0])).T
rank_avg = rank_avg.loc[(classifier_order, augmenter_idx), :]

if WRITE_RESULTS:
    (rank_avg
      .style.highlight_min(axis=0, props="font-weight:bold;")
      .format(lambda val: "{:.1f}".format(val) if not np.isnan(val) else '')
      .to_latex('{}/rank_avg_combined_excluding_CiFRUS.txt'.format(FIGURE_SAVEDIR),
                convert_css = True,
                sparse_index = False,
                hrules = True))

#%% Figure 9: Distribution of the differences between metric scores

# -----------------------------------------------------------------------------
# Figure options
classifiers = ['GNB', 'KNN', 'LR', 'MLP', 'DT', 'ADB', 'RF'] # GNB | KNN | LR | MLP | DT | ADB | RF
augmenter_name = cifrus_names[2] # 0 | 1 | 2
# -----------------------------------------------------------------------------


fig, axes = plt.subplots(ncols = len(metric_avg.columns), nrows = len(classifiers),
                         figsize = (14, len(classifiers)*1.5), sharey = False, sharex = False,
                         squeeze = False)

for j, (classifier) in enumerate(classifiers):
    for i, (metric) in enumerate(metric_avg.columns):
        ax = axes[j, i]
        df = metric_avg[metric].loc[classifier, :].unstack(level = 0)
        df = df.loc[augmenter_order, :]
        a = df.iloc[0, :]
        b = df.loc[augmenter_name, :]   
        pval = ttest_rel(a, b)[1]
        lim = (b - a).abs().max() * 1.2
        ax.hist(b - a, color = 'C' + str(i))
        
        ax.set_xlim(-lim, lim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
        if j == 0:
            ax.set_title(r'$\Delta$ ' + metric)
        if i == axes.shape[1] - 1:
            ax.text(1.05, 0.5, classifier, rotation = 90, ha = 'left', va = 'center', transform = ax.transAxes)
        ax.axvline(x = 0, linestyle = '-.', color = '#000', linewidth = 0.5)
        ax.text(0.98, 0.8, 'p = {:.2f}'.format(pval), ha = 'right', transform = ax.transAxes)

plt.subplots_adjust(wspace = 0.4, hspace = 0.8)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("number of datasets")
filename = 'metric_diff_between_{}_and_{}_for_classifiers'.format(a.name,
                                                                 'CiFRUS')
savefig(filename)
plt.show()

#%% Figure 6: Imbalanced and sample-scarce dataset with performance difference

# -----------------------------------------------------------------------------
# Figure options
metric = ['AUC', 'Kappa', 'F1-score', 'balanced-accuracy', 'MCC']
threshold = 0.005
# -----------------------------------------------------------------------------


n_minority_samples = dataset_info['samples'] * dataset_info['% minority class'] / 100
n_features = dataset_info['features']

cond = pd.concat({'imbalanced': dataset_info['% minority class'] <= 25,
                'sparse': n_minority_samples < 2*n_features},
                axis = 1)
cond = cond[cond.any(axis = 1)]

fig, axes = plt.subplots(nrows = len(metric),
                         ncols = metric_avg.index.levshape[0],
                         sharex = True, sharey = True,
                         figsize = (2.5*metric_avg.shape[1], 1.5*len(metric)), squeeze = False)

for i, classifier in enumerate(metric_avg.index.levels[0]):
    for j, m in enumerate(metric):
        ax = axes[j, i]
        metric_diff = metric_avg.loc[classifier, m].unstack(level = 0).loc[:, cond.index].T
        metric_diff = metric_diff - metric_diff['Baseline'].values[:, None]
        metric_diff = metric_diff.drop('Baseline', axis = 1)
        
        df = pd.concat([cond, metric_diff], axis = 1)
        
        count_up = (metric_diff >= threshold).sum(axis = 0)
        count_down = (metric_diff < -threshold).sum(axis = 0)
        
        count = pd.concat({'$ \geq' + str(threshold) + '$': count_up,
                           '$ \leq -' + str(threshold) + '$': count_down}, axis = 1)
        
        count = count.loc[augmenter_order[1:], :]
        count = count.rename(index = {'CiFRUS (balanced-train/test)': 'CiFRUS (bal.-train/test)'})
        count.plot.bar(ax = ax, legend = False, zorder = 2)
        
        if i == axes.shape[1] - 1:
            ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], title = 'Post-augmentation\n{} change'.format(m))
        ax.grid(axis = 'y', alpha = 0.5)
        if j == 0:
            ax.set_title(classifier)
        ax.set_xlabel(None)
plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Augmentation", labelpad = 150)
plt.ylabel("Number of datastes")
plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
savefig('metric_diff_imbalanced_sample_scarce_{}'.format('_'.join(metric)))
plt.show()

#%% Figure 8, 10: Augmentation methods ranked by both average and variance of metric

# -----------------------------------------------------------------------------
# Figure options
## To generate Figure 8:
classifier = ['RF', 'ADB', 'LR'] 
## To generate Figure 10:
#classifier = ['GNB', 'KNN', 'MLP', 'DT'] 
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(ncols = metrics.shape[1], nrows = len(classifier),
                         sharey = True, sharex = True, squeeze = False,
                         figsize = (13, 3.5*len(classifier)))

for i, clf in enumerate(classifier):
    for j, metric in enumerate(metrics.columns):
        ax = axes[i, j]
        
        results = {}
        colname_avg = 'Avg'
        colname_var = 'Var'
        legend_title = 'Mean rank'
        
        # average, variance for all datasets
        df = metrics.loc[clf, metric]
        var = df.groupby(level = ['dataset', 'augmentation'], sort = False).var().unstack(level = 0)
        avg = df.groupby(level = ['dataset', 'augmentation'], sort = False).mean().unstack(level = 0)
        rank_var = var.rank(axis = 0).prod(axis = 1) ** (1/var.shape[1])
        rank_avg = (-avg).rank(axis = 0).prod(axis = 1) ** (1/avg.shape[1])
        results['All'] = pd.concat({colname_avg: rank_avg, colname_var: rank_var}, axis = 1)
        
        # average, variance for imbalanced/sample-sparse
        n_minority_samples = dataset_info['samples'] * dataset_info['% minority class'] / 100
        n_features = dataset_info['features']
        
        cond = pd.concat({'imbalanced': dataset_info['% minority class'] <= 25,
                        'sparse': n_minority_samples < 2*n_features},
                       axis = 1)
        cond = cond[cond.any(axis = 1)]
        
        dataset_ids = cond.index
        var = df.loc[cond.index].groupby(level = ['dataset', 'augmentation'], sort = False).var().unstack(level = 0)
        avg = df.loc[cond.index].groupby(level = ['dataset', 'augmentation'], sort = False).mean().unstack(level = 0)
        rank_var = var.rank(axis = 0).prod(axis = 1) ** (1/var.shape[1])
        rank_avg = (-avg).rank(axis = 0).prod(axis = 1) ** (1/avg.shape[1])
        results['Imbalanced\n or sample-scarce'] = pd.concat({colname_avg: rank_avg, colname_var: rank_var}, axis = 1)
        results = pd.concat(results, names = ['Datasets', 'Augmentation'])
        
        df = results.unstack(level = 0).reorder_levels([1, 0], axis = 1).sort_index(level = 0, axis = 1)
        df = df.loc[augmenter_order, :]
        df = df.rename(index = {'CiFRUS (balanced-train/test)': 'CiFRUS (bal.-train/test)'})
        sns.heatmap(df, annot = True, ax = ax, cbar = False,
                          fmt = ".2f", cmap = 'viridis_r',
                          vmin = 3, vmax = 7)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticklabels(df.columns.get_level_values(1), rotation = 0)
        
        ax.vlines(df.columns.levshape[1], -1, df.shape[0]+1, color = 'w', linewidth = 2)
        if i == 0:
            for d, title in enumerate(df.columns.levels[0]):
                ax.text((df.columns.levshape[1])*d + df.columns.levshape[1] / 2, -0.75,
                        title,  ha = 'center', va = 'center')

        ax.tick_params(left= False, bottom = False)
        if i == len(classifier) - 1:
            ax.set_xlabel('{} rank'.format(metric))
        if j > 0:
            ax.set_ylabel('')
        if j == axes.shape[1] - 1 and len(classifier) > 1:
            ax.text(1.05, 0.5, clf, rotation = 90, ha = 'left', va = 'center', transform = ax.transAxes)

plt.subplots_adjust(wspace = 0.4, hspace = 0.8)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("augmentation", labelpad = 120)
plt.subplots_adjust(wspace = 0.1, hspace = 0.05)
savefig('average_variance_all_metrics{}'.format('_{}'.format(classifier[0]) if len(classifier) == 1 else ''))
plt.show()
