#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:00:57 2018
Coupling layer file.f
@author: l.faury
"""

from mlp import mlp
import tensorflow as tf
from mask import EvenMask
import numpy as np


def create_coupling_layer(mode, intensor, mask, depth, width, counter):
    return CouplingLayer(mode, intensor, mask, depth, width, counter)


class CouplingLayer(object):
    ''' Affine coupling layer class '''

    def __init__(self, mode, intensor, mask, depth, width, counter):
        ''' Init
        Args:
            mode: string, 'ff' (feed-forward mode) or 'fb' (feed-backward mode)
            intensor: tensor like, input tensor of the coupling layer
            mask: binary mask object
            depth: int, depth of the MLPs s(•) and t(•)
            width: int, width of the MLPs s(•) and t(•)
            counter: int, number of the coupling layer
        '''
        self.mode = mode
        self.size = int(intensor.get_shape()[1])
        self.maskval = mask()
        self.ctr = counter
        self.depth = depth
        self.width = width
        self.build(intensor)

    def build(self, x):
        ''' Build the coupling layers 
        Args:
            x: tensor like, input tensor
        '''
        name = 'coupling_'+str(self.ctr)
        with tf.variable_scope(name):
            # mask
            self.mask = tf.Variable(self.maskval, dtype=tf.float64, trainable=False)

            # input tensor
            xpp = tf.identity(x)
            self.xd = tf.multiply(xpp, self.mask)

            # defining s(•) and t(•) (MLPs)
            self.s = mlp(0, self.xd, self.size, self.depth, self.width, tf.nn.tanh)
            self.t = mlp(1, self.xd, self.size, self.depth, self.width)

            # feed forward output
            if (self.mode == 'ff'):
                self.xa = tf.multiply(xpp, tf.exp(self.s.y)) + self.t.y
                self.y = self.xd + tf.multiply(self.xa, 1-self.mask)
            # feed backward output
            elif (self.mode == 'fb'):
                self.xa = tf.multiply(tf.exp(-self.s.y), xpp-self.t.y)
                self.y = self.xd + tf.multiply(self.xa, 1-self.mask)
            else:
                raise ValueError('Inexistent mode for coupling layer')

            # jacobian determinant computation for the forward function (closed form)
            if (self.mode == 'ff'):
                self.detjac = tf.exp(tf.reduce_sum(
                    tf.multiply(1-self.mask, self.s.y), axis=1))
                self.detjactest = tf.matrix_determinant(
                    tf.stack([tf.gradients(self.y[:, idx], x)[0] for idx in range(self.size)], axis=1))
            elif (self.mode == 'fb'):
                self.detjac = tf.exp(-tf.reduce_sum(
                        tf.multiply(1-self.mask, self.s.y), axis=1))
                self.detjactest = tf.matrix_determinant(
                    tf.stack([tf.gradients(self.y[:, idx], x)[0] for idx in range(self.size)], axis=1))
            else:
                raise ValueError('Inexistent mode for coupling layer')


################################################################################
#                             UNIT TESTING                                     #
################################################################################

class TestCouplingLayer(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(0)
        np.random.seed(0)
        self.z = tf.placeholder(dtype=tf.float64, shape=[None, 5])
        self.x = tf.placeholder(dtype=tf.float64, shape=[None, 5])
        self.mask = EvenMask(5)
        self.inputs = [[-0.5, 0.0, +0.5, +1.0, -1.0]]
        self.outputs = [[-0.5, 0.04952787824980896, 0.5, 1.0248171122708212, -1.0]]

    def test_forward(self):
        # init
        clff = create_coupling_layer(mode='ff', intensor=self.z, mask=self.mask, depth=5, width=128, counter=0)

        # act
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            x_out = session.run(clff.y, feed_dict={self.z: self.inputs})

        # check
        self.assertListEqual(self.outputs[0], list(x_out[0]))

    def test_backward(self):
        # init
        #clff = create_coupling_layer(mode='ff', intensor=self.z, mask=self.mask, depth=5, width=128, counter=0)
        clfb = create_coupling_layer(mode='fb', intensor=self.x, mask=self.mask, depth=5, width=128, counter=0)

        # act
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            #x_out = session.run(clff.y, feed_dict={self.z: self.inputs})
            z_out = session.run(clfb.y, feed_dict={self.x: self.outputs})

        # check
        self.assertListEqual(self.inputs[0], list(z_out[0]))

    def test_roundtrip(self):
        # init
        clff = create_coupling_layer(mode='ff', intensor=self.z, mask=self.mask, depth=5, width=128, counter=0)
        clfb = create_coupling_layer(mode='fb', intensor=self.x, mask=self.mask, depth=5, width=128, counter=0)

        # act
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            x_out = session.run(clff.y, feed_dict={self.z: self.inputs})
            z_out = session.run(clfb.y, feed_dict={self.x: x_out})

        # check
        self.assertListEqual(self.inputs[0], list(z_out[0]))


    def test_jacobian(self):
        # init
        clfb = create_coupling_layer(mode='fb', intensor=self.x, mask=self.mask, depth=5, width=128, counter=0)

        # act
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            jac, jactest = session.run([clfb.detjac, clfb.detjactest], feed_dict={self.x: self.outputs})

        # check
        self.assertLessEqual(np.max(np.abs(jac - jactest)), 0)

    def test_all(self):
        clff = create_coupling_layer(mode='ff', intensor=self.z, mask=self.mask, depth=5, width=128, counter=0)
        clfb = create_coupling_layer(mode='fb', intensor=self.x, mask=self.mask, depth=5, width=128, counter=0)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # testing feed forward
            yff = session.run(clff.y, feed_dict={self.z: self.inputs})
            print('Feed forward test \n -----------')
            print(yff)

            # testing feed backward
            print('\n Feed backward test,expects 0\n -----------')
            zfb = session.run(clfb.y, feed_dict={self.x: yff})
            print(np.max(np.abs(np.round(10 ** 6 * (self.inputs - zfb)))))

            # testing jacobian
            print('\n Jacobian test, expects 0\n -----------')
            jac, jactest = session.run([clfb.detjac, clfb.detjactest], feed_dict={self.x: yff})
            print(np.max(np.abs(jac - jactest)))


if __name__ == '__main__':
    tf.test.main()

