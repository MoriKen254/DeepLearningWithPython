#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import random
import sys

sys.path.append('../util')
from RandomGenerator import Uniform
from ActivationFunction import Sigmoid, Tanh, ReLU

class HiddenLayer:

    def __init__(self, dim_input_signal, dim_output_signal, weights, biases, rand_obj, activation):

        if rand_obj is None:
            rand_obj = random()

        self.biases = [0] * dim_output_signal

        if weights is None:
            weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
            w = 1./dim_input_signal

            for j in range(dim_output_signal):
                for i in range(dim_input_signal):
                    random_generator = Uniform(-w, w)
                    weights[j][i] = random_generator.compute(rand_obj)

        if biases is None:
            self.biases = [0] * dim_output_signal

        self.dim_input_signal = dim_input_signal
        self.dim_output_signal = dim_output_signal
        self.weights = weights
        self.biases = biases
        self.rand_obj = rand_obj

        # Configure Activation function
        if activation == 'Sigmoid' or activation == None:
            self.activation = Sigmoid()
        elif activation == 'Tanh':
            self.activation = Tanh()
        elif activation == 'ReLU':
            self.activation = ReLU()
        else:
            raise ValueError('specified activation function "' + activation + '" is not supported')

    def output(self, input_signals):
        pre_activations = [0] * self.dim_output_signal

        # E = Sum{ w^T * x + b }
        for j, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            for w, input_signal in zip(weight, input_signals):
                pre_activations[j] += w * input_signal

            pre_activations[j] += bias # linear output

        return pre_activations

    def foward(self, input_signals):
        return output(input_signals)

