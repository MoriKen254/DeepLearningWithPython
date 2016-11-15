#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import random

class HiddenLayer:


    def __init__(self, dim_input_signal, dim_output_signal, weights, baiases, rand_obj, activation):

        if rand_obj is None:
            rand_obj = random()

        self.rand_obj = rand_obj

        self.dim_input_signal = dim_input_signal
        self.dim_output_signal = dim_output_signal

        self.biases = [0] * dim_output_signal

        if weights is None:
            self.weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
            w = 1./dim_input_signal

            for j in range(dim_output_signal):
                for i in range(dim_input_signal):
                    weights[j][i] = 0
