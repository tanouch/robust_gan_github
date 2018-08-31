#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:28:04 2018
Creates toy training distributions
@author: l.faury
"""

from abc import ABC
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy


class Distribution(ABC):
    ''' Base distribution class '''

    def __init__(self, dimx):
        ''' Init
        Args:
            dimx: int, x dimension
        '''
        self.dimx = dimx
        self._build()

    def _build(self):
        raise NotImplementedError()

    def create_dataset(self, size, tt_ratio):
        ''' Creates the corresponding training dataset (classifier)
        Args:
            size: int, size of dataset
            tt_ratio: float in [0,1], train/test ratio
        '''
        raise NotImplementedError()

    def get_batch(self, s):
        ''' Returns a batch of size s from the dataset
        Args:
            s: int, batch size
        '''
        return NotImplementedError()

    def knn(self, X, k):
        ''' k-Nearest neighbor function between x and self.Xtrain
        Args:
            X: np.array, input features
            k: int
        Returns:
            np.array= knn labels based on (self.X, self.Y)
        '''
        nbrs = NearestNeighbors(n_neighbors=k,
                                algorithm='auto').fit(self.Xtrain)
        _, idx = nbrs.kneighbors(X)
        y = (np.mean(self.Ytrain[idx], axis=1)) >= 0.5
        return y.flatten()


class GaussianMixtureModel(Distribution):
    ''' GMM distribution class '''

    def __init__(self, dimx):
        super().__init__(dimx)

    def _build(self):
        self.n_components = 16
        self.labels = np.random.randint(0, 2, self.n_components)
        self.mixture_weights = np.random.uniform(1, 10, self.n_components)
        self.mixture_weights /= np.sum(self.mixture_weights)
        self.means = np.random.uniform(-10, 10, (self.n_components, self.dimx))
        self.covar = []
        for _ in range(self.n_components):
            self.covar.append(np.random.uniform(1, 2)*np.eye(self.dimx))

    def create_dataset(self, size, tt_ratio=0.5):
        X = np.zeros((size, self.dimx))
        Y = np.zeros((size,), dtype=int)
        num_samples = 0
        while (num_samples < size):
            component = np.random.choice(a=np.arange(self.n_components),
                                         size=1,
                                         replace=True,
                                         p=self.mixture_weights)[0]
            mean = self.means[component, :]
            covar = self.covar[component]
            x = mean + np.dot(scipy.linalg.sqrtm(covar),
                              np.random.normal(0, 1, (self.dimx,)))
            y = self.labels[component]
            X[num_samples, :] = x
            Y[num_samples] = y
            num_samples += 1
        self.X = X
        self.Y = Y
        train_size = int(tt_ratio*size)
        self.Xtrain = self.X[:train_size, :]
        self.Ytrain = self.Y[:train_size]
        self.Xtest = self.X[train_size:, :]
        self.Ytest = self.Y[train_size:]


class RandomHypercube(Distribution):
    ''' Random Hypercube distribution (NIPS 2003 challenge) '''

    def __init__(self, dimx):
        super().__init__(dimx)

    def _build(self):
        self.n_feature = self.dimx
        self.n_informative = self.dimx
        self.n_redundant = self.n_feature - self.n_informative
        self.n_clusters = np.random.randint(1, self.n_feature+1)

    def create_dataset(self, size, tt_ratio=0.5):
        self.X, self.Y = make_classification(n_samples=size,
                                             n_features=self.n_feature,
                                             n_informative=self.n_informative,
                                             n_redundant=self.n_redundant,
                                             n_clusters_per_class=self.n_clusters)
        train_size = int(tt_ratio*size)
        self.Xtrain = self.X[:train_size, :]
        self.Ytrain = self.Y[:train_size]
        self.Xtest = self.X[train_size:, :]
        self.Ytest = self.Y[train_size:]


def create_distribution(name, dim):
    ''' Creates and return a distribution object
    Args:
        name: string
    '''
    if name == 'random_hypercube':
        distribution = RandomHypercube(dim)
    elif name == 'gmm':
        distribution = GaussianMixtureModel(dim)
    else:
        return NotImplementedError()
    return distribution
