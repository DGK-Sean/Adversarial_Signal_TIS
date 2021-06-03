#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:35:02 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ddx_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):
    return np.tanh(x)


def ddx_tanh(x):
    return 1 - tanh(x)*tanh(x)


def relu(x):
    res = []
    for item in x:
        if item > 0:
            res.append(item)
        else:
            res.append(0)
    return np.asarray(res)


def ddx_relu(x):
    res = []
    for item in x:
        if item > 0:
            res.append(1)
        else:
            res.append(0)
    return np.asarray(res)


def leaky_relu(x, alpha):
    res = []
    for item in x:
        if item > 0:
            res.append(item)
        else:
            res.append(item*alpha)
    return np.asarray(res)


def ddx_leaky_relu(x, alpha):
    res = []
    for item in x:
        if item > 0:
            res.append(1)
        else:
            res.append(alpha)
    return np.asarray(res)


def softplus(x):
    return np.log(1+np.exp(x))


def ddx_softplus(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # evenly sampled time at 200ms intervals
    input_range = np.arange(-3., 3., 0.025)
    out_sigmoid = sigmoid(input_range)
    out_tanh = tanh(input_range)
    out_relu = relu(input_range)
    out_lrelu = leaky_relu(input_range,0.2)
    out_softplus = softplus(input_range)

    out_ddx_sigmoid = ddx_sigmoid(input_range)
    out_ddx_tanh = ddx_tanh(input_range)
    out_ddx_relu = ddx_relu(input_range)
    out_ddx_lrelu = ddx_leaky_relu(input_range,0.2)
    out_ddx_softplus = ddx_softplus(input_range)

    # --- Functions ---
    # Fig size
    figPres = plt.figure(figsize=(8, 6))
    axPres  = figPres.add_subplot(111)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    # Axis lines
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    relu_pos = len(input_range)//2
    # Lines
    plt.plot(input_range, out_sigmoid, '-', linewidth=2, c='#bca72d')
    plt.plot(input_range[relu_pos:], out_relu[relu_pos:], '-', linewidth=5, c='darkblue', alpha=1)
    plt.plot(input_range[:relu_pos+1], out_relu[:relu_pos+1], '-', linewidth=2, c='darkblue', alpha=1)
    plt.plot(input_range, out_tanh, '-', linewidth=2, c='#258458')
   # plt.plot(input_range, out_lrelu, '-', linewidth=2, c='red', alpha=1)
   # plt.plot(input_range, out_softplus, '-', linewidth=2, c='orange', alpha=1)

    # Legend
    p_sig = mpatches.Patch(facecolor='#bca72d', label='Sigmoid', lw=1, edgecolor='black')
    p_rel = mpatches.Patch(facecolor='darkblue', label='ReLU', lw=1, edgecolor='black')
    p_tan = mpatches.Patch(facecolor='#258458', label='Tanh', lw=1, edgecolor='black')
    #p_lrel = mpatches.Patch(facecolor='red', label='Leaky ReLU', lw=1, edgecolor='black')
    #p_soft = mpatches.Patch(facecolor='orange', label='Softplus', lw=1, edgecolor='black')
    plt.legend(handles = [p_sig, p_tan, p_rel], loc='upper left')

    # Plot properties, axis etc
    plt.gca().yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.2)
    plt.gca().xaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.2)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], rotation='horizontal', fontsize=13)
    plt.yticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], rotation='horizontal', fontsize=13)
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])

    axPres.yaxis.set_label_coords(-0.06,0.475)
    axPres.yaxis.tick_right()
    plt.xlabel('x', fontsize=17)
    plt.ylabel('f(x)', fontsize=17, rotation='horizontal')

    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)

    plt.savefig('Activation.png')
    plt.show()

    # --- Derivatives ---
    # Fig size
    figPres = plt.figure(figsize=(8, 6))
    axPres  = figPres.add_subplot(111)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    # Axis lines
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)

    # Lines
    plt.plot(input_range, out_ddx_softplus, '-', linewidth=2, c='orange')
    plt.plot(input_range[relu_pos+1:], out_ddx_relu[relu_pos+1:], '-', linewidth=5, c='darkblue', alpha=1)
    plt.plot(input_range[:relu_pos+1], out_ddx_relu[:relu_pos+1], '-', linewidth=2, c='darkblue', alpha=1)
    plt.plot(input_range[:relu_pos+1], out_ddx_lrelu[:relu_pos+1], '-', linewidth=2, c='red', alpha=1)
    #plt.plot(input_range, out_ddx_lrelu, '-', linewidth=2, c='red', alpha=1)
    plt.plot(input_range, out_ddx_tanh, '-', linewidth=2, c='#258458')
    plt.plot(input_range, out_ddx_sigmoid, '-', linewidth=2, c='#bca72d')
    plt.plot(input_range[relu_pos+1:], out_ddx_lrelu[relu_pos+1:], '-', linewidth=2, c='red', alpha=1)


    # Legend
    p_sig = mpatches.Patch(facecolor='#bca72d', label='Sigmoid', lw=1, edgecolor='black')
    p_rel = mpatches.Patch(facecolor='darkblue', label='ReLU', lw=1, edgecolor='black')
    p_tan = mpatches.Patch(facecolor='#258458', label='Tanh', lw=1, edgecolor='black')
    p_lrel = mpatches.Patch(facecolor='red', label='Leaky ReLU', lw=1, edgecolor='black')
    p_soft = mpatches.Patch(facecolor='orange', label='Softplus', lw=1, edgecolor='black')
    plt.legend(handles = [p_sig, p_tan, p_rel, p_lrel, p_soft], loc='upper left')

    # Plot properties, axis etc
    plt.gca().yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.2)
    plt.gca().xaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.2)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], rotation='horizontal', fontsize=13)
    plt.yticks([-0.5, 0, 0.5, 1, 1.5],[-0.5, 0, 0.5, 1, 1.5], rotation='horizontal', fontsize=13)
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-0.5, 1.5])
    axPres.yaxis.set_label_coords(-0.06,0.475)
    axPres.yaxis.tick_right()
    plt.xlabel('x', fontsize=17)
    plt.ylabel('f \'(x)', fontsize=17, rotation='horizontal')

    plt.savefig('Activation_Derivative.png')
    plt.show()








