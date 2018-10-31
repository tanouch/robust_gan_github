#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:25:57 2018

@author: l.faury
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from abc import ABC, abstractmethod



class Classifier(ABC):
    ''' Base classifier class '''
    def __init__(self, input_dim, class_no, lr):
        ''' Init
        Args:
            input_dim: int, feature space dimension
            class_no: int, number of class
            lr: float, learnin rate
        '''
        self.fdim = input_dim
        self.class_no = class_no
        self.lr = lr
        self._build()

    @abstractmethod
    def _build(self):
        raise NotImplementedError()

    def train(self, session, X, Y, bsz, iters):
        ''' Train the classifier
        Args:
            session: tensorflow session
            X: np.array, input features
            Y: np.array, labels
            bsz: int, batch size
            iters: int, number of iterations
        '''
        size = np.shape(X)[0]
        assert (bsz <= size), "Invalid batch size %i" % bsz
        for _ in range(iters):
            idx = np.random.choice(a=np.arange(0, size),
                                   size=bsz,
                                   replace=False)
            x = X[idx, :]
            y = np.reshape(Y[idx], (-1, 1))
            feed_dict = {self.x: x, self.t: y}
            session.run(self.fit, feed_dict=feed_dict)

    def get_accuracy(self, session, X, Y):
        ''' Evaluate accuracy on (X,Y)
        Args:
            session: tensorflow session
            X: np.array, input features
            Y: np.array, labels
        '''
        Y = np.reshape(Y, (-1, 1))
        feed_dict = {self.x: X, self.t: Y}
        accuracy = session.run(self.accuracy, feed_dict=feed_dict)
        return accuracy


class MlpBinaryClassifier(Classifier):
    ''' MLP Binary Classifier '''
    def __init__(self, input_dim, depth, width, facq, lr, l2_param, gp_param, name='bcmlp'):
        self.depth = depth
        self.width = width
        self.facq = facq
        self.name = name
        self.l2_param = l2_param
        self.gp_param = gp_param
        self.epsilons = np.logspace(-1, 0.5, 20)
        super().__init__(input_dim, 2, lr)

    

    def _predict(self, x):
        with tf.name_scope('layers'):
            net = tf.identity(x)
            for _ in range(1, self.depth-1):
                net = slim.fully_connected(net, self.width,
                                            activation_fn=self.facq,
                                            weights_regularizer=slim.l2_regularizer(self.l2_param))
            net = slim.fully_connected(net, 1, activation_fn=None,
                                        weights_regularizer=slim.l2_regularizer(self.l2_param))
            return net

    def _build(self):
        with tf.name_scope(self.name+'_placeholder'):
            # placeholders
            self.x = tf.placeholder(dtype=tf.float64,
                                    shape=[None, self.fdim],
                                    name='input')
            self.t = tf.placeholder(dtype=tf.float64,
                                    shape=[None, 1],
                                    name='label')

        # model
        self.y = self._predict(self.x)

        with tf.name_scope('ops'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.t,
                                                                    logits=self.y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.add_gradient_penalty()
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.fit = self.optimizer.minimize(self.loss)
            self.pred_labels = tf.round(tf.nn.sigmoid(self.y))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels, self.t),
                                                    tf.float64))
            self.gradients = tf.gradients(self.loss, self.x)[0]
            self.sign_gradients = tf.sign(self.gradients)

    def add_gradient_penalty(self):
        if (self.gp_param>0.0):
            self.GPgradients = tf.gradients(self.y, [self.x])[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.GPgradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
            self.loss += self.gp_param * self.gradient_penalty

    
    def _train(self, session, X, Y, distribution, iters):
        self.train(session=session,
                      X=X,
                      Y=Y,
                      bsz=20,
                      iters=iters)
        training_accuracy = self.get_accuracy(session,
                                            distribution.Xtrain,
                                            distribution.Ytrain)
        test_accuracy = self.get_accuracy(session,
                                        distribution.Xtest,
                                        distribution.Ytest)
        
        epsilon_accs = list()
        for epsilon in self.epsilons:
            adv_x = self.get_adv_x(session, distribution.Xtest, distribution.Ytest, epsilon)
            epsilon_accuracy = self.get_accuracy(session,
                                            adv_x,
                                            distribution.Ytest)
            #print("epsilon=%g, adv_acc=%g" % (epsilon, epsilon_accuracy))
            epsilon_accs.append(epsilon_accuracy)

        msg = 'Training accuracy: %f \t Testing accuracy: %f' \
                '\t epsilon accuracy: %f' % \
                            (training_accuracy, test_accuracy, 
                            epsilon_accs[-3])
        print(msg)
        return epsilon_accs

            
    def get_adv_x(self, session, X, Y, epsilon):
        Y = np.reshape(Y, (-1, 1))
        feed_dict = {self.x: X, self.t: Y}
        adv_x = session.run(self.x + epsilon*self.sign_gradients, feed_dict=feed_dict)
        return adv_x
                



