from abc import ABC, abstractmethod

import numpy as np


class SamplingWithOracleCache(ABC):
    def __init__(self, dim, n):
        """
        :param dim: dimension of the samples
        :param n: maximum number of samples to store
        """
        self.dim = dim
        self.n = n
        self.samples = np.zeros((n, dim))
        self.fevals = np.zeros((n))
        self.fgrads = np.zeros((n, dim))
        self.pdf = None
        self.is_weights = np.ones((n))

    @abstractmethod
    def update_history(self, samples, fevals, fgrads, pdf, is_weights):
        pass

    @abstractmethod
    def sample_batch_using_history(self, samples, fevals, fgrads, pdf):
        pass


class ImportanceMixingBase(SamplingWithOracleCache):
    def __init__(self, dim, n, alpha):
        super().__init__(dim, n)
        self.alpha = alpha

    def _proba_to_keep_new(self, x, new_pdf, old_pdf):
        ''' Computes the probability to keep a point of new distribution '''
        return np.maximum(self.alpha*np.ones(x.shape[0]), 1 -
                          np.exp(old_pdf(x) - new_pdf(x)))

    def _proba_to_keep_old(self, x, new_pdf, old_pdf):
        ''' Computes the probability to keep a point of old distribution '''
        return np.exp(np.minimum(new_pdf(x) - old_pdf(x) + np.log(1-self.alpha), 0))

    def _sample_from_new_distribution(self, n, pdf, sampling_fun):
        i = 0
        new_samples = np.zeros((n, self.dim))
        old_pdf = self.pdf
        while i < n:
            x = sampling_fun(1)
            proba_keep = self._proba_to_keep_new(x, pdf, old_pdf)
            keep = np.random.binomial(1, proba_keep)
            if keep:
                new_samples[i] = x
                i += 1
        return new_samples


class ImportanceMixingWithRejection(ImportanceMixingBase):
    def __init__(self, dim, n, alpha):
        super().__init__(dim, n, alpha)

    def update_history(self, samples, fevals, fgrads, pdf, is_weights):
        self.samples = samples
        self.fgrads = fgrads
        self.fevals = fevals
        self.pdf = pdf
        self.is_weights = is_weights

    def sample(self, nevals, sampling_fun, pdf, popsize):
        if nevals == 0:
            newx = sampling_fun(popsize)
            oldx = np.empty((0, self.dim))
            fvals = np.array([])
            dfvals = np.empty((0, self.dim), float)
            is_weights = np.array([])
        else:
            newx, oldx, fvals, dfvals, is_weights = \
                self.sample_batch_using_history(self.n, sampling_fun, pdf)
        return newx, oldx, fvals, dfvals, is_weights

    def sample_batch_using_history(self, n_samples, sampling_fun, pdf):
        samples, fevals, fgrad, is_weights = self._sample_from_history(pdf)
        n_new_samples = n_samples - samples.shape[0]
        new_samples = self._sample_from_new_distribution(n_new_samples, pdf, sampling_fun)
        return new_samples, samples, fevals, fgrad, is_weights

    def _sample_from_history(self, pdf):
        x_old = self.samples
        old_pdf = self.pdf
        proba_keep = self._proba_to_keep_old(x_old, pdf, old_pdf)
        keep = np.random.binomial(1, proba_keep)
        idx = np.where(keep)
        return x_old[idx], self.fevals[idx], self.fgrads[idx], self.is_weights[idx]


class ImportanceMixingWithWeighting(ImportanceMixingBase):
    def __init__(self, dim, n, alpha):
        super().__init__(dim, n, alpha)
        self.index = 0

    def update_history(self, samples, fevals, fgrads, pdf, is_weights):
        n_batch = samples.shape[0]
        self.samples[:n_batch] = samples
        self.fevals[:n_batch] = fevals
        self.fgrads[:n_batch] = fgrads
        self.is_weights[:n_batch] = is_weights
        self.pdf = pdf
        self.index = n_batch

    def sample_batch_using_history(self, n_samples, sampling_fun, pdf):
        samples, fevals, fgrads, is_weights = self._sample_from_history(pdf)
        new_samples = self._sample_from_new_distribution(n_samples, pdf, sampling_fun)
        return new_samples, samples, fevals, fgrads, is_weights

    def _sample_from_history(self, pdf):
        x_old = self.samples[:self.index]
        old_pdf = self.pdf
        proba_keep = self._proba_to_keep_old(x_old, pdf, old_pdf)
        return x_old, self.fevals[:self.index], self.fgrads[:self.index], proba_keep * self.is_weights[:self.index]

    def sample(self, nevals, sampling_fun, pdf, popsize):
        if nevals == 0:
            newx = sampling_fun(popsize)
            oldx = np.empty((0, self.dim))
            fvals = np.array([])
            dfvals = np.empty((0, self.dim), float)
            is_weights = np.array([])
        else:
            newx, oldx, fvals, dfvals, is_weights = \
                self.sample_batch_using_history(popsize, sampling_fun, pdf)
        return newx, oldx, fvals, dfvals, is_weights

def create_sampler(name, dim, alpha, n_fevals, n_batch):
    if name == 'mixing_with_rejection':
        return ImportanceMixingWithRejection(dim, n_batch, alpha)
    if name == 'mixing_with_weighting':
        return ImportanceMixingWithWeighting(dim, n_fevals, alpha)
    return None
