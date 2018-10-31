#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:29:46 2018

@author: l.faury
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_graph(path, ax, scatter, distribution, list_of_f, i):
    colors = ['g', 'b', 'c', 'k']
    h_list = list()
    for k, f in enumerate(list_of_f):
        cs, h = mesh_vizu(ax, f, colors[k], 
                        xlim=[np.min(distribution.Xtrain[:, 0]), np.max(distribution.Xtrain[:, 0])],
                        ylim=[np.min(distribution.Xtrain[:, 1]), np.max(distribution.Xtrain[:, 1])])
        h_list.append(h[0])
    ax.legend(h_list, ['Base Classifier', 'L2Reg Classifier', 'GP Classifier', 'Robust Classifier'])
    plt.savefig(path+'fig_'+str(i))
    scatter.remove()
    for coll in cs.collections:
        coll.remove()

def mesh_vizu(ax, f, col, xlim, ylim, scope=0):
    """ Plot a contour plot of function f
    Args:
        function: functions to plot contours for
        xlim : [-x,x]
        ylim : [-y,y] limits the plot
        scope = string, name
    Returns:
        ax
    """
    x, y = np.mgrid[xlim[0]:xlim[1]:0.2, ylim[0]:ylim[1]:0.2]
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = f(np.array([[x[i, j], y[i, j]]]))
    cs = ax.contour(x, y, z, colors=col, levels=[0.5])
    h,_ = cs.legend_elements()
    return cs, h


def plot_eps_acc(path, epsilons, base_mlp_eps_accs, robust_mlp_eps_acc, i):
    plt.figure(figsize=(15, 7))
    plt.plot(epsilons, base_mlp_eps_accs, 'r', label="Base Classifier")
    plt.plot(epsilons, robust_mlp_eps_acc, 'b', label="Robust Classifier")
    plt.legend(loc=4)
    plt.savefig(path+'eps_acc_'+str(i))
    plt.close()


def plot_eps_acc_final(path, epsilons, base_mlp_eps_accs, robust_mlp_eps_acc):
    plt.figure(figsize=(15, 7))
    time = np.arange(len(robust_mlp_eps_acc))
    for i, epsilon in enumerate(epsilons[:5]):
        plt.plot(time, robust_mlp_eps_acc[:,i], label=str(round(epsilon, 1)))
    plt.legend(loc=4)
    plt.savefig('final_eps_acc_wrt_time')
    plt.close()