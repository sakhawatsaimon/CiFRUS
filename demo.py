#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:35:18 2024

@author: Sakhawat
"""

from cifrus.cifrus import CiFRUS
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

##----------------------
## Load dataset
##----------------------

X, y = datasets.load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


##----------------------
## CiFRUS augmentation
##----------------------

cfrs = CiFRUS()
## Fit and augment the train split
X_train_resampled, y_train_resampled = cfrs.fit_resample(X_train, y_train)


##----------------------
## Classifier training
##----------------------

clf = RandomForestClassifier()
clf.fit(X_train_resampled, y_train_resampled)

##----------------------
## Ensemble prediction
##----------------------

# CiFRUS takes the classifier's predict_proba() function as parameter
# X_test is resampled, and the probabilities are aggregated internally
y_pred = cfrs.resample_predict_proba(clf.predict_proba, X_test)