#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 2023

@author: Sakhawat
"""
import numpy as np
#import pandas as pd
from scipy.stats import mode

class CiFRUS():
    """
    Use the sorted feature vectors from the training dataset to find the
    insertion location i for each feture value in a query sample, then use a
    radius (i +- k) to uniformly sample synthetic values.
    k : (int) Radius of draw range
    expand_range: (bool) if a new instance is out of bounds in X_train and
        expand_range is set to True, then expand the sampling bounds B_l and B_u
        (for given query instance only)
    random_state : (int) controls the randomness of the uniform sampling, default = None
    """
    def __init__(self, k = None, expand_range = True, random_state = None):
        if k is None:
            # heuristic function, default
            self._k_func = lambda X: min(20, int(np.sqrt(X.shape[0])))
        elif callable(k):
            # user-defined callable (must accept X_train as argument)
            self._k_func = k
        else:
            # user-defined integer
            self._k_func = lambda X: k
        self.expand_range = expand_range
        self.random_state = random_state
        self.default_synthesis_rate = 5
            
    def _set_random_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def _searchsorted2d(self, X):
        """
        Find insert location of each feature value in each query instance in X
        into sorted X_train
        """
        return np.array([np.searchsorted(self.X_train_sorted.T[i], X.T[i]) for i in np.arange(X.shape[1])]).T
    
    def fit(self, X_train):
        self.X_train_sorted = np.sort(X_train.copy(), axis = 0)
        self.N = self.X_train_sorted.shape[0]
        self.m = self.X_train_sorted.shape[1]
        self.k = int(self._k_func(X_train))
    
    def resample(self, X, Y = None, r = None, include_parents = True):
        """
        Create r synthetic instances for each instance in a set of query instances
        
        Parameters
        ----------
        X : Numpy 1D or 2D array
            Query instance(s) for which synthetic instances should be generated.
        Y : Numpy 1D array, optional
            Class labels for X. The default is None (class labels unknown).
        r : int, optional
            Number of synthetic instances per real instance (synthesis rate).
            The default is self.default_synthesis_rate
        include_parents : bool, optional
            If True, then X is returned with synthetic instances. The default
            is True.
        
        Returns
        -------
        Xv : Numpy 2D array
            Augmented instances (can also include X).
        Yv : Numpy 1D array, optional
            Class lables associated with Xv. Only returned if Y is not None.
        """     
        if r is None:
            r = self.default_synthesis_rate
        # if X is 1D, convert to 2D
        X = X.reshape(-1, X.shape[-1])
        assert X.shape[1] == self.m
        
        # find the insertion location for each feature value in each query instance
        i = self._searchsorted2d(X)
        lower_range_idx = i - self.k
        upper_range_idx = i + self.k - 1
        
        # ensure indices are within bounds
        lower_range_idx[lower_range_idx < 0] = 0
        lower_range_idx[lower_range_idx >= self.N] = self.N - 1
        upper_range_idx[upper_range_idx < 0] = 0
        upper_range_idx[upper_range_idx >= self.N] = self.N - 1
        
        # B is the collection of uniform sampling bounds for each feature
        # B[0] is B_l (lower bound), B[1] is B_u (upper bound)
        B = self.X_train_sorted[[lower_range_idx, upper_range_idx],
                                         range(self.m)]
        if self.expand_range:
            B[0] = np.minimum(B[0], X)
            B[1] = np.maximum(B[1], X)
            
        self._set_random_seed()
        Xv = np.random.uniform(low = B[0],
                               high = B[1],
                               size = [r, X.shape[0], self.m])
        Xv = np.moveaxis(Xv, 0, 1)
        
        if include_parents:
            Xv = np.concatenate([X.reshape(X.shape[0], -1, X.shape[-1]), Xv], axis = 1)
        Xv = Xv.reshape(-1, X.shape[-1])
        
        if Y is None:
            return Xv
        Yv = np.tile(Y, [r + int(include_parents), 1]).T.reshape(-1)
        return Xv, Yv
    
    def resample_balanced(self, X, Y, r_majority = None, include_parents = True):
        """
        Create synthetic instances for each instance in a set of query instances
        while balancing classes. The synthesis rate for each minority class is
        inferred such that after augmentation, the number of instances in each
        minority class is greater than or equal to the number of instances in
        the majority class.

        Parameters
        ----------
        X : Numpy 1D or 2D array
            Query instance(s) for which synthetic samples should be generated.
        Y : Numpy 1D array
            Class labels for X.
        r_majority : int, optional
            Number of synthetic instances for the majority class. The default is None
            which implies r_majority = self.default_synthesis_rate
        include_parents : bool, optional
            If True, then X is returned with synthetic instances. The default
            is True.

        Returns
        -------
        Xv : Numpy 2D array
            Augmented instances (can also include X).
        Yv : Numpy 1D array
            Class lables associated with Xv.
        """
        
        if r_majority is None:
            r_majority = self.default_synthesis_rate
            
        i = int(include_parents)
        labels, counts = np.unique(Y, return_counts = True)
        majority_label_count = max(counts)
        majority_augmented_count = max(1, r_majority + i) * majority_label_count
        
        proportions = {k: v for k, v in zip(labels, counts)}
        R = {k: r_majority if v == majority_label_count\
                                   else int(np.ceil((majority_augmented_count - v*i)/ v)) \
            for k, v in proportions.items()}
        Xv, Yv = [], []
        for label, ri in R.items():
            mask = Y == label
            Xi, Yi = self.resample(X[mask, :], Y[mask], r = ri,
                                   include_parents = include_parents)
            Xv.append(Xi)
            Yv.append(Yi)
        Xv = np.vstack(Xv)
        Yv = np.hstack(Yv)
        return Xv, Yv
    
    def resample_proportional(self, X, Y, R, include_parents = True):
        """
        Create synthetic instances from query instances with the synthesis rate
        for each class defined by the user.

        Parameters
        ----------
        X : Numpy 1D or 2D array
            Query instance(s) for which synthetic samples should be generated.
        Y : Numpy 1D array
            Class labels for X.
        R : Union[dict,list,np.ndarray]
            Synthesis rate each class in Y. Values must be int.
        include_parents : bool, optional
            If True, then X is returned with synthetic instances. The default
            is True.

        Returns
        -------
        Xv : Numpy 2D array
            Generated synthetic instances (can also include X).
        Yv : Numpy 1D array
            Class lables associated with Xv.
        """
        labels = np.unique(Y)
        assert len(R) == len(labels)
        if isinstance(R, list) or isinstance(R, np.ndarray): 
            R = {k: v for k, v in zip(labels, R)}
        #elif isinstance(R, pd.Series):
        #    R = R.to_dict()
        assert all([isinstance(v, (int, np.integer)) for v in R.values()])
            
        Xv, Yv = [], []
        for label, n in R.items():
            mask = Y == label
            # ni is the number of 
            Xi, Yi = self.resample(X[mask, :], Y[mask], n = n,
                                  include_parents = include_parents)
            Xv.append(Xi)
            Yv.append(Yi)
        Xv = np.vstack(Xv)
        Yv = np.hstack(Yv)
        return Xv, Yv
    
    def fit_resample(self, X, Y, r = None, include_parents = True, balanced = True):
        """
        Fit and augment the training dataset

        Parameters
        ----------
        X : Numpy 1D or 2D array
            Query instance(s) for which synthetic instances should be generated.
        Y : Numpy 1D array, optional
            Class labels for X. The default is None (class labels unknown).
        r : int, optional
            Number of synthetic instances per real instance (synthesis rate).
            The default is self.default_synthesis_rate
        include_parents : bool, optional
            If True, then X is returned with synthetic instances. The default
            is True.
        balanced : bool, optional
            If True, then perform balanced augmentation. The default is True.

        Returns
        -------
        Xv : Numpy 2D array
            Augmented instances (can also include X).
        Yv : Numpy 1D array
            Class lables associated with Xv.
        """
        self.fit(X)
        if balanced:
            return self.resample_balanced(X, Y, r, include_parents)
        return self.resample(X, Y, r, include_parents)
    
    def resample_predict_proba(self, func_predict_proba, X, r = None, include_parents = True):
        """
        Predict probability of test instances by augmenting the instance,
        predicting the probability of the augmented instaces, and computing the
        mean probability.

        Parameters
        ----------
        func_predict_proba : callable
            scikit-learn API compatible predict_proba function that accepts a set
            of instances as argument.
        X : Numpy 1D or 2D array
            Query instance(s) with unknown class label.
        r : int, optional
            Number of synthetic instances per real instance (synthesis rate).
            The default is self.default_synthesis_rate
        include_parents : bool, optional
            If True, then X is included with the synthetic instances. The default
            is True.

        Returns
        -------
        scores: array-like of shape (n_instances, n_classes)
            The probability of each instance for each class in the model.

        """        
        nx = 1 if len(X.shape) == 1 else len(X)
        # Augment the test instances
        Xv = self.resample(X, Y = None,
                           r = r,
                           include_parents = include_parents)        
        # Predict probabilities for each synthetic instance
        scores = func_predict_proba(Xv)
        
        # Note: at present, real instances are implicitly grouped with the
        # synthetic ones in all resample methods. If Xv is shuffled prior to
        # return, or random sampling is done on synthetic instances to achieve
        # exact split of classes, then the current reshape-based
        # implementation should be replaced with instance-by-instance
        # augmentation and prediction.
        
        # Reshape scores to align each real instance with associated synthetic instances
        scores = scores.reshape(nx, -1, scores.shape[-1])
        # Take column mean to get mean score for each real instance
        return scores.mean(axis = 1)
    
    def resample_predict(self, func_predict, X, r = None, include_parents = True):
        """
        Predict class labels of test instances by augmenting the instance,
        predicting the class labels of the augmented instaces, and majority
        voting.

        Parameters
        ----------
        func_predict : callable
            scikit-learn API compatible predict function that accepts a set
            of instances as argument.
        X : Numpy 1D or 2D array
            Query instance(s) with unknown class label.
        r : int, optional
            Number of synthetic instances per real instance (synthesis rate).
            The default is self.default_synthesis_rate
        include_parents : bool, optional
            If True, then X is included with the synthetic instances. The default
            is True.

        Returns
        -------
        y_pred : ndarray of shape (n_instances,)

        """  
        nx = 1 if len(X.shape) == 1 else len(X)
        Xv = self.resample(X, Y = None,
                          r = r,
                          include_parents = include_parents)
        # Predict class labels for each augmented instance
        y_pred = func_predict(Xv)
        # Reshape labels to align each real instance and associated synthetic instances
        
        # Note: at present, real instances are grouped with the
        # synthetic ones in all resample methods. If Xv is shuffled prior to
        # return, or random sampling is done on synthetic instances to achieve
        # exact split of classes, then the current implementation, which
        # implicitely assumes that synthetic instances are grouped with their
        # parents, should be replaced.
        
        # Reshape pred labels to align each real instance with associated
        # synthetic instances
        y_pred = y_pred.reshape(nx, -1)
        # majority voting decides class label
        return mode(y_pred)[0]
        
        
        
        