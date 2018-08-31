#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:59:16 2018

@author: l.faury
"""

from abc import ABC, abstractmethod
import tensorflow as tf


class GradientEstimator(ABC):
    ''' A gradient estimator class '''

    def __init__(self, generator, lr, bsz, name):
        ''' Initialization
        Args:
            generator: InvGen object, generator to train
            lr: int, learning rate
            bsz: int, batch size
            name: string
        '''
        self.name = name
        self.bsz = bsz
        self.build_estimator(generator)
        self.build_optimizer(lr)

    @abstractmethod
    def build_estimator(self, generator):
        raise NotImplementedError()

    def build_optimizer(self, lr):
        ''' Builds the optimization ops'''
        with tf.name_scope(self.name+'_optim'):
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
            self.optimize = self.optimizer.apply_gradients(zip(self.gradient_estimator,
                                                               self.params))


class ScoreFunctionEstimator(GradientEstimator):
    ''' Score Function Estimator Class '''

    def __init__(self, generator, lr, bsz, name='sf'):
        super().__init__(generator, lr, bsz, name)

    def build_estimator(self, generator):
        self.params = generator.params
        with tf.name_scope(self.name+'_estim'):
            self.fitn = tf.placeholder(tf.float64, [None, ],
                                       name='fitn')
            # unnormalized gradient
            self.gradient_estimator_N = tf.gradients(generator.llkh,
                                                     self.params,
                                                     self.fitn)
            # normalize
            self.gradient_estimator = list(map(lambda x: tf.div(x, self.bsz),
                                               self.gradient_estimator_N))


class PathwiseDerivativeEstimator(GradientEstimator):
    ''' Pathwise Derivative Estimator Class '''
    def __init__(self, generator, lr, bsz, name='pd'):
        super().__init__(generator, lr, bsz, name)

    def build_estimator(self, generator):
        self.params = generator.params
        with tf.name_scope(self.name+'_estim'):
            self.fgrad = tf.placeholder(tf.float64, [None, generator.dim],
                                        name='fgrad')
            # unnormalized gradient
            self.gradient_estimator_N = tf.gradients(generator.x,
                                                     self.params,
                                                     self.fgrad)
            # normalize
            self.gradient_estimator = list(map(lambda x: tf.div(x, self.bsz),
                                               self.gradient_estimator_N))


class LogLikelihoodEstimator(GradientEstimator):
    ''' Log-likelihood training '''
    def __init__(self, generator, lr, bsz, name='llog'):
        super().__init__(generator, lr, bsz, name)

    def build_estimator(self, generator):
        self.params = generator.params
        with tf.name_scope(self.name+'_estim'):
            self.loss = -tf.reduce_mean(generator.llkh, axis=None)

    def build_optimizer(self, lr):
        with tf.name_scope(self.name+'_optim'):
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.optimize = self.optimizer.minimize(self.loss)


def create_estimator(gd_estimator, generator, lr, bsz=0):
    ''' Creates a gradient estimator
    Args:
        gd_estimator: string, name of the gradient estimator
        generator: InvGen object, generator to train
        lr: float, learnin rate
        bsz: int, batch size
    '''
    if gd_estimator not in ['pd', 'sf', 'natural_sf', 'natural_pd', 'llog']:
        raise ValueError('Unknowwn gradient estimator')
    elif gd_estimator == 'sf':
        gradient_estimator = ScoreFunctionEstimator(generator, lr, bsz)
    elif gd_estimator == 'pd':
        gradient_estimator = PathwiseDerivativeEstimator(generator, lr, bsz)
    elif gd_estimator == 'llog':
        gradient_estimator = LogLikelihoodEstimator(generator, lr, bsz)
    return gradient_estimator


#        # natural gradient update
#        if 'natural' in gd_estimator:
#            self.natural_gd_estimator = [tf.placeholder(
#                    dtype=tf.float64,
#                    shape=p.shape.as_list()) for p in self.params]