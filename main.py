#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:50:19 2018
Robust GAN script.
@author: l.faury
"""

import argparse
import pprint as pp
import distribution as distrib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import utils as ut

from classifier import MlpBinaryClassifier
sys.path.append('./invgen')
from invgen import InvNet
from matplotlib import gridspec
from estimator import create_estimator
import os

def main(args):
    # reproducibility
    # ---------------
    path = str(args['seed'])+"/"
    if (not os.path.isdir(path)):
        os.mkdir(path)

    np.random.seed(args['seed'])
    tf.set_random_seed(args['seed'])
    # ---------------

    # parameters
    # ---------------
    dim = args['dim']
    distrib_name = args['distribution']
    data_size = args['data_size']
    # ---------------

    # create dataset
    # ---------------
    # TODO train/tests split
    distribution = distrib.create_distribution(name=distrib_name,
                                               dim=dim)
    distribution.create_dataset(data_size)

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    ax.scatter(distribution.Xtrain[:, 0],
               distribution.Xtrain[:, 1],
               c=distribution.Ytrain,
               cmap='bwr')
    # ---------------

    with tf.Session() as session:
        # create invnet/gan
        # ---------------
        invnet = InvNet(dim=dim,
                        depth=args['inv_depth'],
                        name='invnet')
        invnet_cpy = InvNet(dim=dim,
                            depth=args['inv_depth'],
                            name='invnet_copy')

        llog_optimizer = create_estimator(gd_estimator='llog',
                                          generator=invnet,
                                          lr=args['gen_lr'])

        pd_optimizer = create_estimator(gd_estimator='pd',
                                        generator=invnet_cpy,
                                        lr=args['gen_lr'],
                                        bsz=1000)
        # ---------------
        # create classifier
        # ---------------
        mlp = MlpBinaryClassifier(input_dim=dim,
                                  depth=args['classif_depth'],
                                  width=args['classif_width'],
                                  facq=tf.nn.tanh,
                                  lr=args['classif_lr'],
                                  beta=0.0,
                                  name='rcmlp')

        reg_mlp = MlpBinaryClassifier(input_dim=dim,
                                  depth=args['classif_depth'],
                                  width=args['classif_width'],
                                  facq=tf.nn.tanh,
                                  lr=args['classif_lr'],
                                  beta=1.,
                                  name='bcmlp')

        base_mlp = MlpBinaryClassifier(input_dim=dim,
                                  depth=args['classif_depth'],
                                  width=args['classif_width'],
                                  facq=tf.nn.tanh,
                                  lr=args['classif_lr'],
                                  beta=0.,
                                  name='bcmlp')
        # ---------------
        # init
        # ---------------
        session.run(tf.global_variables_initializer())
        # ---------------

        # Initial training of invertible network
        # ---------------
        print('Training Invertible Neural Network..')
        train_invnet_llog(session,
                          invnet=invnet,
                          llog_optimizer=llog_optimizer,
                          X=distribution.Xtrain,
                          iters=1000)

        print('Train both classifiers under real distribution')
        base_mlp_eps_accs = base_mlp._train(session=session,
                      X=distribution.Xtrain,
                      Y=distribution.Ytrain,
                      distribution=distribution,
                      iters=30000)
        reg_mlp_eps_accs = reg_mlp._train(session=session,
                      X=distribution.Xtrain,
                      Y=distribution.Ytrain,
                      distribution=distribution,
                      iters=30000)
        print("")

        # decision function
        def f_cl(cl):
            def f(x):
                x = np.reshape(x, [-1, dim])
                feed_dict = {cl.x: x}
                return session.run(cl.y, feed_dict)
            return f

        # Optimization
        # ---------------
        invnet_cpy.copy(session, invnet)
        robust_mlp_eps_acc = list()
        for i in range(15):
            # sample points from invnet
            x_invnet, _ = invnet_cpy.sample(session, 1000)
            y_invnet = distribution.knn(x_invnet, 10)
            scatter = ax.scatter(x_invnet[:, 0], x_invnet[:, 1],
                                 c=y_invnet, alpha=0.2, cmap='bwr')

            # train classifier under invnet distribution
            print('Train robust classifier under invnet distribution')
            accs = mlp._train(session=session,
                      X=x_invnet,
                      Y=y_invnet,
                      distribution=distribution,
                      iters=2500)
            robust_mlp_eps_acc.append(accs)

            base_f, reg_f, robust_f = f_cl(base_mlp), f_cl(reg_mlp), f_cl(mlp)
            ut.plot_graph(path, ax, scatter, distribution, [base_f, reg_f, robust_f], i)
            
            # adversarial training of the invnet
            print("Train invnet..")
            invnet_cpy.copy(session, invnet)
            train_invnet_pd(session=session,
                            invnet=invnet_cpy,
                            disc=mlp,
                            distribution=distribution,
                            pd_optimizer=pd_optimizer,
                            iters=10)
            print("")

        ut.plot_eps_acc(path, mlp.epsilons, base_mlp_eps_accs, accs, i)                
        ut.plot_eps_acc_final(path, mlp.epsilons, base_mlp_eps_accs, np.array(robust_mlp_eps_acc))
        # ---------------

def train_invnet_llog(session, invnet, llog_optimizer, X, iters):
    ''' Train the invnet for MLE objective
    Args:
        session: tensorflow session
        invnet: InvNet object
        llog_optimizer= Estimator object
        X: np.array, dataset
    '''
    for _ in range(iters):
        feed_dict = {invnet.xfb: X}
        session.run(llog_optimizer.optimize, feed_dict=feed_dict)


def train_invnet_pd(session, invnet, disc, pd_optimizer, distribution, iters):
    ''' Train the invnet with pathwise-derivative
    Args:
        session: tensorflow session
        invnet: InvNet object
        disc: Classifier object
        distribution: Distribution object
        pd_optimizer= Estimator object
    '''
    for j in range(iters):
        x, z = invnet.sample(session, pd_optimizer.bsz)
        y = np.reshape(distribution.knn(x, 10), (-1, 1))
        feed_dict_disc = {disc.x: x, disc.t: y}
        disc_grads = -session.run(disc.gradients, feed_dict=feed_dict_disc)
        feed_dict_invnet = {invnet.zff: z, pd_optimizer.fgrad: disc_grads}
        session.run(pd_optimizer.optimize, feed_dict=feed_dict_invnet)

def train_residual_gan(session, invnet, disc, pd_optimizer, distribution, iters):
    ''' Train the invnet with pathwise-derivative
    Args:
        session: tensorflow session
        invnet: InvNet object
        disc: Classifier object
        distribution: Distribution object
        pd_optimizer= Estimator object
    '''
    for j in range(iters):
        x, z = invnet.sample(session, pd_optimizer.bsz)
        y = np.reshape(distribution.knn(x, 10), (-1, 1))
        feed_dict_disc = {disc.x: x, disc.t: y}
        disc_grads = -session.run(disc.gradients, feed_dict=feed_dict_disc)
        feed_dict_invnet = {invnet.zff: z, pd_optimizer.fgrad: disc_grads}
        session.run(pd_optimizer.optimize, feed_dict=feed_dict_invnet)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Black-Box Optimization Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dim',
                        help='Problem dimension',
                        default=2,
                        type=int)
    parser.add_argument('--distribution',
                        help='Problem distribution',
                        default='gmm',
                        type=str)
    parser.add_argument('--inv_depth',
                        help='Depth of the invertible network',
                        default=5,
                        type=int)
    parser.add_argument('--classif_depth',
                        help='Depth of the MLP classifier',
                        default=4,
                        type=int)
    parser.add_argument('--classif_width',
                        help='Width of the MLP classifier',
                        default=64,
                        type=int)
    parser.add_argument('--gen_lr',
                        help='Generator learning rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--classif_lr',
                        help='Classifier learning rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--data_size',
                        help='Size of dataset',
                        default=2000,
                        type=int)
    parser.add_argument('--seed',
                        help='Random seed',
                        default=1,
                        type=int)

    args = vars(parser.parse_args())
    pp.pprint(args)
    main()