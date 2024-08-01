#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:39:15 2023

@author: Tanzira
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Imbalanced-learn:
# https://imbalanced-learn.org/stable/install.html#getting-started    
# Geometric SMOTE:
# https://geometric-smote.readthedocs.io/en/latest/install.html
# -----------------------------------------------------------------------------

# Scoring metrics
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             balanced_accuracy_score,
                             cohen_kappa_score,
                             matthews_corrcoef)

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# Augmentation algorithms
from cifrus import CiFRUS
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import (RandomOverSampler,
                                    SMOTE,
                                    BorderlineSMOTE,
                                    SVMSMOTE,
                                    ADASYN)
from gsmote import GeometricSMOTE

from sklearn.model_selection import StratifiedKFold

from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Utility function
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):,}h {int(minutes)}m {int(seconds)}s"


# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------

# Randomized algorithm seed (to ensure reproducibility)
SEED = 2023

# Overwrite existing files in the results directory if set to True
OVERWRITE_PREVIOUS = False

# Only iterate over each dataset and do not train models if set to True
DRY_RUN = True

# Output directory for results
OUTPUT_DIR = Path("./results/performance")

# Estimate the best k as an integer, or a function of X_train
# If none, use the default specified in CiFRUS
k_func = None

# Number of synthetic samples per real sample
# Used for CiFRUS (train), CiFRUS (train/test)
r = 10

# Number of synthetic samples per real sample of the majority class
# Used for CiFRUS (balanced-train/test)
r_majority = 5

# Number of splits for cross-validation
n_splits = 10

# Number of repeated cross-validation runs
n_repeats = 10

# Probability threshold for determining class label (for MCC, F1-score, ...)
p_threshold = 0.5


# SVC was not included due to long runtime
classifiers = {'GNB': GaussianNB(),
               'KNN': KNeighborsClassifier(),
               #'SVC': SVC(kernel = 'linear', random_state = SEED, probability=True),
               'LR': LogisticRegression(random_state = SEED),
               'MLP': MLPClassifier(solver='lbfgs', random_state = SEED),
               'DT': DecisionTreeClassifier(random_state = SEED),
               'ADB': AdaBoostClassifier(random_state = SEED),
               'RF': RandomForestClassifier(random_state = SEED)
               }

augmenters = {'ROS': RandomOverSampler(random_state = SEED),
              'RUS': RandomUnderSampler(random_state = SEED),
              'SMOTE': SMOTE(random_state = SEED),
              'Borderline-SMOTE': BorderlineSMOTE(random_state = SEED),
              'SVM-SMOTE': SVMSMOTE(random_state = SEED),
              'ADASYN': ADASYN(random_state = SEED),
              'G-SMOTE': GeometricSMOTE(random_state = SEED)
              }

cifrus_names = ['CiFRUS (train)',
                'CiFRUS (train/test)',
                'CiFRUS (balanced-train/test)']

# Each metric function takes 3 params (all 3 may not be needed)
# to maintain a common interface that can be looped over
metrics = {'AUC': lambda y_true, y_pred, scores: \
                           roc_auc_score(y_true, scores),
           'F1-score': lambda y_true, y_pred, scores: \
                           f1_score(y_true, y_pred),
           'Kappa': lambda y_true, y_pred, scores: \
                           cohen_kappa_score(y_true, y_pred),
           'balanced-accuracy': lambda y_true, y_pred, scores: \
                           balanced_accuracy_score(y_true, y_pred),
           'MCC': lambda y_true, y_pred, scores: \
                           matthews_corrcoef(y_true, y_pred)}
# -----------------------------------------------------------------------------


# For parallel execution, no need to change if running a single python instance
try:
    total_nodes, node_id = int(sys.argv[1]), int(sys.argv[2])
    print('New node with nnodes={}, offset={}'.format(total_nodes, node_id))
except:
    print('Running single node')
    total_nodes, node_id = 1, 0

# Read dataset info
basepath = Path('./data/')
dataset_names = pd.read_csv(Path(basepath, 'dataset_names.txt'),
                            header = None, quotechar = "'").squeeze()
dataset_info = pd.DataFrame(index = range(1, len(dataset_names)+1),
                            columns = ['name', 'samples', 'features'])
subdirs = sorted([d for d in basepath.iterdir() if d.is_dir()])

Path(OUTPUT_DIR).mkdir(parents = True, exist_ok = True)

for d in subdirs:
    dataset_idx = int(d.name)
    dataset_name = dataset_names[dataset_idx-1]
    dataset_info.loc[dataset_idx, :] = dataset_name, \
        *loadmat(d.joinpath('data.mat'))['X'].shape
    
dataset_info_arr = dataset_info.sort_values('samples').reset_index().to_numpy()
#%%
# -----------------------------------------------------------------------------

# Utility function
def metrics_from_scores(scores):
    """ Calculate performance metrics from scores """
    metric_df = pd.DataFrame(index = scores.columns,
                             columns = metrics.keys(),
                             dtype = float)
    y_true = scores['Y_true']
    for metric_name, metric_func in metrics.items():
        for augmenter_name in metric_df.index:
            scores_augmenter = scores[augmenter_name]
            y_pred = (scores[augmenter_name] >= p_threshold).astype(int)
            metric_df.loc[augmenter_name, metric_name] = metric_func(y_true,
                                                                     y_pred,
                                                                     scores_augmenter)
    metric_df = metric_df.drop('Y_true')
    return metric_df
    
starttime = time.time()
# Loop over each dataset
for i, (dataset_idx, dataset_name, n, m) in enumerate(dataset_info_arr):
    dataset_starttime = time.time()
    if i % total_nodes != node_id:
        continue
    mat = loadmat(basepath.joinpath('{}/data.mat'.format(dataset_idx)))
    X = mat['X']
    Y = mat['y'].ravel()
    Y = pd.factorize(Y, sort = True)[0]
    
    out_path = Path("{}/{}.csv".format(OUTPUT_DIR, dataset_name))
    
    print(f'Node {node_id} picked up {dataset_name} ({n=}, {m=})')
    if out_path.is_file() and not OVERWRITE_PREVIOUS:
        print("\t{} results exist, skipping".format(dataset_name))
        continue
    
    if DRY_RUN:
        continue
    
    results = {}
    
    for it in range(n_repeats):
        print(f'repeat={it}')
        # iteration is used as cv seed to get a different split each time
        cv = StratifiedKFold(n_splits = n_splits, shuffle = True,
                             random_state = it)
        results_clf = {}
        
        for classifier_name, clf in classifiers.items():
            print(f'\t{classifier_name=}')
         
            scores = []
            for train_index, test_index in cv.split(X, Y):
                X_train, Y_train = X[train_index], Y[train_index]
                X_test, Y_test = X[test_index], Y[test_index]
                scores_fold = pd.DataFrame(Y_test, columns = ['Y_true'])
                
                clf.fit(X_train, Y_train)
                scores_fold['Baseline'] = clf.predict_proba(X_test)[:, -1]
                
                for augmenter_name, augmenter in augmenters.items():
                    try:
                        X_train_v, y_train_v = augmenter.fit_resample(X_train, Y_train)
                        clf.fit(X_train_v, y_train_v)
                        scores_fold[augmenter_name] = clf.predict_proba(X_test)[:, 1]
                    except Exception as e:
                        if augmenter_name == 'ADASYN':
                            # ADASYN fails to augment if classes are balanced
                            # Assume no augmentation was done
                            scores_fold[augmenter_name] = scores_fold['Baseline'] 
                        else:
                            print(e)
                            scores_fold[augmenter_name] = np.nan
                
                cifrus = CiFRUS(k = k_func, random_state = SEED)
                cifrus.fit(X_train)
                
                # Augmented training
                X_train_v, Y_train_v = cifrus.resample(X_train, Y = Y_train,
                                                       r = r)
                clf.fit(X_train_v, Y_train_v)
                # Baseline prediction: CiFRUS (train)
                scores_fold[cifrus_names[0]] = clf.predict_proba(X_test)[:, 1]
                # Ensemble prediction: CiFRUS (train/test)
                scores_fold[cifrus_names[1]] = cifrus.resample_predict_proba(clf.predict_proba,
                                                                             X_test,
                                                                             r = r)[:, 1]
                
                # Augmented (balanced) training
                X_train_v, Y_train_v = cifrus.resample_balanced(X_train, Y = Y_train,
                                                                r_majority = r_majority)
                clf.fit(X_train_v, Y_train_v)
                # Ensemble prediction: CiFRUS (balanced-train/test)     
                scores_fold[cifrus_names[2]] = cifrus.resample_predict_proba(clf.predict_proba,
                                                                             X_test,
                                                                             r = r)[:, 1]
                scores.append(scores_fold)
                
            # combine all scores and calculate metrics
            scores = pd.concat(scores).reset_index(drop = True)
            metric_df = metrics_from_scores(scores)

            results_clf[classifier_name] = metric_df
        results_clf = pd.concat(results_clf)
        results[it] = results_clf
    results = pd.concat(results, names = ['iter', 'classifier', 'augmentation'])
    results = results.reorder_levels([1, 0, 2])
    # sort by classifier name
    results = results.loc[classifiers.keys(), :]
    results.to_csv(out_path, float_format = '%.6f')
    try:
        print(results.groupby(['classifier', 'augmentation']).mean()['AUC'].unstack().T)
    except:
        pass
    dataset_endtime = time.time()
    print(f"Dataset time: {format_time(dataset_endtime - dataset_starttime)}")
endtime = time.time()
print(f"Total elapsed time: {format_time(endtime - starttime)}")