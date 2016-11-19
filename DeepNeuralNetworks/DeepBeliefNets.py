#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import csv
import sys
import random

from RestrictedBoltzmannMachines import RestrictedBoltzmannMachines
sys.path.append('../SingleLayerNeuralNetworks')
from LogisticRegression import LogisticRegression
sys.path.append('../MultiLayerNeuralNetworks')
from HiddenLayer import HiddenLayer

class DeepBeliefNets:

    def __init__(self, dim_input_signal, dims_hidden_layers, dim_output_layer, rand_obj, use_csv=False):
        self.use_csv = use_csv

        if rand_obj is None:
            rand_obj = random(1234)
        self.rand_obj = rand_obj

        self.dim_input_signal = dim_input_signal
        self.dims_hidden_layers = dims_hidden_layers
        self.dim_output_signal = dim_output_layer
        self.cnt_layers = dims_hidden_layers.size
        self.sigmoid_layers = []#HiddenLayer(self.cnt_layers)
        self.rbm_layers = []

        # construct multi-layer
        dim_prev_layer_input = 0
        for i, (dim_hidden_layer, sigmoid_layer) in enumerate(self.dims_hidden_layers, self.sigmoid_layers):
            if i == 0:
                dim_curr_layer_input = dim_input_signal
            else:
                dim_curr_layer_input = dim_prev_layer_input

            # construct hidden layers with sigmoid function
            #   weight matrices and bias vectors will be shared with RBM layers
            self.sigmoid_layers.append(HiddenLayer(dim_curr_layer_input, dim_hidden_layer,
                                                   None, None, rand_obj, 'sigmoid'))

            # construct RBM layers
            self.rbm_layers.append(RestrictedBoltzmannMachines(dim_curr_layer_input, dim_hidden_layer,
                                                               sigmoid_layer.weights, sigmoid_layer.biases, None, rand_obj))

            dim_prev_layer_input = dim_hidden_layer

        # logistic regression layer for output
        self.logistic_layer = LogisticRegression(self.dims_hidden_layers[self.cnt_layers-1], self.dim_output_signal)


