#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:33:26 2018
Multi Layer Perceptron class file.
@author: l.faury
"""

import tensorflow as tf
import numpy as np


def dense(x, num_output, counter, ac_fn=None):
    ''' Fully connected layer.
    Args:
        x: tensor like, input of the layer
        num_outputs: int, number of output units of the layer
        counter: int, number of output units
        ac_fn: function, layer activation function
    Returns:
        z = Wx + b
    '''
    name = 'dense_' + str(counter)
    # xavier-glorot initialization variance
    insz = int(x.get_shape()[1])  # input size
    std = np.sqrt(1.0/insz)  # initial uniform noise support

    # declare variables
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', shape=[insz, num_output], dtype=tf.float64,
                            initializer=tf.random_normal_initializer(0, std))
        b = tf.get_variable(
            'b', shape=[num_output], dtype=tf.float64, initializer=tf.constant_initializer(0.))
    z = tf.matmul(x, W)+b
    if ac_fn is not None:
        z = ac_fn(z)
    return z


def dense_wn(x, num_output, counter, ac_fn=None):
    ''' Fully connected layer with weight normalization (Salimans & Kingma 2016)
    Args:
        x: tensor like, input of the layer
        num_outputs: int, number of output units of the layer
        counter: int, number of output units
        ac_fn: function, layer activation function
    Returns:
        z = Wx + b
    '''
    name = 'dense_wn_' + str(counter)
    # xavier-glorot initialization variance
    insz = int(x.get_shape()[1])  # input size
    # TODO: data-based initialization
    std = np.sqrt(1.0/insz)

    # declare variables
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        V = tf.get_variable('V', shape=[insz, num_output], dtype=tf.float64,
                            initializer=tf.random_normal_initializer(0, 0.05))
        g = tf.get_variable('g', shape=[num_output], dtype=tf.float64,
                            initializer=tf.random_normal_initializer(std))
        b = tf.get_variable('b', shape=[num_output], dtype=tf.float64,
                            initializer=tf.constant_initializer(0.))
    z = tf.matmul(x, V)
    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
    z = scaler*z + b
    if ac_fn is not None:
        z = ac_fn(z)
    return z


class mlp(object):
    ''' Multi-Layer Perceptron class.
        Default (and only) activation is leaky ReLU.
    '''

    def __init__(self, counter, intensor, outsize, depth, width, facq=None):
        ''' Init
        Args:
            counter: int, to set the mlp name
            intensor: tensor like, the network input tensor
            outsize: int, the network output size
            depth: int, network depth
            width: int, network width
            facq: function, final activation function
        '''
        self.name = 'mlp_'+str(counter)
        if (outsize <= 0 or depth <= 0 or width <= 0):
            raise ValueError(str(type(self) + ' bad init values.'))
        self.opsz = outsize
        self.dpt = depth
        self.wdt = width
        self.facq = facq
        self.build(intensor)

    def build(self, x):
        ''' Builds the mlp graph
            For now, we use a constant width of the layers, and ReLU activation by default.
            Args:
                x: tensor like, the input tensor of the net
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            actv = tf.identity(x)
            for d, c in enumerate(range(self.dpt)):
                actv = dense(actv, self.wdt, c, ac_fn=tf.nn.relu)
#                actv = dense_wn(actv, self.wdt, c, ac_fn=tf.nn.relu)
            # output
            actv = dense(actv, self.opsz, c+1, ac_fn=self.facq)
            alpha = tf.get_variable('alpha', shape=[], dtype=tf.float64,
                                    initializer=tf.constant_initializer(1.))  # learned scale variable
            self.y = tf.multiply(actv, alpha)


if __name__ == '__main__':
    # instanciating test
    x = tf.placeholder(dtype=tf.float64, shape=[None, 2])
    net = mlp(0, x, 10, 5, 128)
    # prediction test
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        inputs = np.random.normal(0, 1, (10, 2))
        y = session.run(net.y, feed_dict={x: inputs})
        print(y)
        if (y.shape != (10, 10)):
            raise ValueError('Somethings wrong with the mlp')
        else:
            print('Success')
