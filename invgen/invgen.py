#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:34:07 2018
Inversible Generator file.
@author: l.faury
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from coupling import CouplingLayer
from mask import EvenMask, OddMask, RandomMask
import matplotlib.pyplot as plt


class InvGen(ABC):
    ''' Invertible Generator Class (parent class)'''

    def __init__(self, dim, name):
        ''' Init
        Args:
            dim: int, dimension of the input and output vector
            gd_estimator: string, gradient estimator name
        '''
        self.dim = dim
        self.name = name
        self.build_model()
        self.build_statistics()
        self.build_copy_op()

    @abstractmethod
    def build_model(self):
        ''' Builds the inverse function model '''
        raise NotImplementedError()

    @abstractmethod
    def build_statistics(self):
        ''' Builds the likelihood of the model'''
        raise NotImplementedError()

    def build_copy_op(self):
        ''' Builds copy operations '''
        self.copyweights = [tf.placeholder(dtype=tf.float64,
                                           shape=p.shape.as_list()) for p in self.allparams]
        self.copyop = [tf.assign(p, w) for p, w in zip(self.allparams, self.copyweights)]

    def copy(self, session, invgen):
        ''' Copy the parameter of another invertible network into self.
        Used for the importance mixing procedure.
        Args:
            session: Tensorflow session
            invgen: InvGen object
        '''
        copyweights = session.run(invgen.allparams)
        feed_dict = {cpy: cp for (cpy, cp) in zip(self.copyweights, copyweights)}
        session.run(self.copyop, feed_dict=feed_dict)

    def sample(self, session, size):
        ''' Samples data points from latent vectors
        Args:
            session: tensorflow session
            size: int, number of samples
        Returns:
            x: array [numpoints x dimension], data vectors
            z: array [numpoints x dimension], corresponding latent vectors
        '''
        z = np.random.normal(0, 1, (size, self.dim))
        feed_dict = {self.zff: z}
        x = session.run(self.x, feed_dict=feed_dict)
        return x, z


class InvNet(InvGen):
    ''' Inversible Neural Network '''

    def __init__(self, dim, depth, name='InvNet'):
        ''' Inits
        Args:
            dim: int, dimension of the input and output vector
            depth: int, number of coupling layers
            name: string
        '''
        self.dpt = depth
        super().__init__(dim, name)

    def build_model(self):
        ''' Builds the inverse function model '''
        with tf.variable_scope(self.name):
            # placeholders
            self.zff = tf.placeholder(dtype=tf.float64, shape=(
                    None, self.dim))  # latent vector,feed-forward mode
            self.xfb = tf.placeholder(dtype=tf.float64, shape=(
                    None, self.dim))  # data vector,feed-backward mode

            # building the inversible generator
            # alternating random masks for the coupling layers
            # fixed neural networks depth (=3) and width (=64)
            # first feed-forward
            hff = tf.identity(self.zff)
            self.masks = []
            self.clff = []  # coupling layers
            for d in range(self.dpt):
                if d % 2 == 0:
                    m = RandomMask(self.dim)
                else:
                    m = RandomMask(self.dim, m.inverse())
                self.masks.append(m)
                if d == 0:
                    self.clff.append(CouplingLayer('ff', hff, m, 3, 64, d))
                else:
                    self.clff.append(CouplingLayer('ff', self.clff[d-1].y, m, 3, 64, d))
            self.x = tf.identity(self.clff[d].y)  # output, data vector

            # then feed-backward
            hfb = tf.identity(self.xfb)
            self.clfb = []  # coupling layers
            for d in range(self.dpt):
                m = self.masks[self.dpt-1-d]
                if d == 0:
                    self.clfb.append(CouplingLayer('fb', hfb, m, 3, 64, self.dpt-1-d))
                else:
                    self.clfb.append(CouplingLayer(
                            'fb', self.clfb[d-1].y, m, 3, 64, self.dpt-1-d))
            self.z = tf.identity(self.clfb[d].y)  # output, latent vector

        # trainable params
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.params_dim = int(np.sum([np.prod(p.shape.as_list()) for p in self.params]))
        # all params
        self.allparams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def build_statistics(self):
        ''' Builds the likelihood of the model'''
        # jacobian determinant
        self.detjac = tf.reduce_prod(
            [self.clfb[c].detjac for c in range(self.dpt)], axis=0)
        self.detjactest = tf.matrix_determinant(tf.stack(
            [tf.gradients(self.z[:, idx], self.xfb)[0] for idx in range(self.dim)], axis=1))
        # log-likelihood
        self.llkh = -0.5*tf.norm(self.z, axis=1)**2 + tf.log(self.detjac)
        # score
        self.dscore = tf.gradients(self.llkh, self.params)


class Gaussian(InvGen):
    ''' Gaussian search distribution class '''

    def __init__(self, dim, name='gaussian'):
        super().__init__(dim, name)

    def build_model(self):
        ''' Builds the inverse function model for the Gaussian search '''
        with tf.variable_scope(self.name):
            # placeholders
            self.zff = tf.placeholder(dtype=tf.float64, shape=(
                    None, self.dim))  # latent vector,feed-forward mode
            self.xfb = tf.placeholder(dtype=tf.float64, shape=(
                    None, self.dim))  # data vector,feed-backward mode

            # covariance square root and mean
            self.A = tf.Variable(np.eye(self.dim), dtype=tf.float64)
            self.S = tf.matmul(self.A, self.A, transpose_a=False,
                               transpose_b=True)
            self.mu = tf.Variable(np.zeros(self.dim,), dtype=tf.float64)

            # feed-forward
            self.x = self.mu + tf.matmul(self.zff, self.A)

            # feed-backward
            self.z = tf.matmul(self.xfb-self.mu, tf.linalg.inv(self.A))  # careful

            # params
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.params_dim = int(np.sum([np.prod(p.shape.as_list()) for p in self.params]))
            # all params
            self.allparams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def build_statistics(self):
        # log-likelihood
        self.llkh = -0.5*tf.linalg.norm(self.z, axis=1)**2 / (tf.linalg.det(self.A)+1e-6)
        # score
        self.dscore = tf.gradients(self.llkh, self.params)


if __name__ == '__main__':
    gen = InvNet(2, 6, 1, 'sf')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # testing sampling
        print('Testing sampling \n --------')
        zff = np.random.normal(0, 1, (1000, 2))
        feed_dict = {gen.zff: zff}
        xs = session.run(gen.x, feed_dict=feed_dict)
        plt.scatter(zff[:, 0], zff[:, 1], c='blue')
        plt.scatter(xs[:, 0], xs[:, 1], c='red')
        plt.show()

        # testing invertibility
        print('\n Testing inversion, expects 0 \n --------')
        feed_dict = {gen.xfb: xs}
        zfb = session.run(gen.z, feed_dict=feed_dict)
        print(np.max(10**6*np.abs((zff-zfb))))

        # testing detjac
        print('\n Testing jacobian, expects 0s \n --------')
        djac, djactest = session.run(
            [gen.detjac, gen.detjactest], feed_dict={gen.xfb: xs})
        print(np.max(np.abs(djac-djactest)))

        # testing log-likelihood
        print('\n Testing log-likelihood \n --------')
        l = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(l, l)
        x = np.transpose(np.array([X.flatten(), Y.flatten()]))
        feed_dict = {gen.xfb: x}
        llkdx, detjac = session.run([gen.llkh, gen.detjac], {gen.xfb: x})
        lkdx = np.reshape(np.exp(llkdx), (100, 100))
        plt.figure()
        CS = plt.contour(X, Y, lkdx, levels=[
                         0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.scatter(xs[:, 0], xs[:, 1], c='red', marker='+')
        plt.show()
