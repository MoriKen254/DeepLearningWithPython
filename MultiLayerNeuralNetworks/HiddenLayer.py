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
from ActivationFunction import Softmax, Step

class HiddenLayer:

    def __init__(self, dim_input_signal, dim_output_signal, weights, biases, rand_obj, activation):

        if rand_obj is None:
            rand_obj = random()

        self.biases = [0] * dim_output_signal

        if weights is None:
            self.weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
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

        if activation == 'sigmoid' or activation == None:
            self.activation = Step()
