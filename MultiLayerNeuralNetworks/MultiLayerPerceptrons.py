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

from HiddenLayer import HiddenLayer

sys.path.append('../SingleLayerNeuralNetworks')
from LogisticRegression import LogisticRegression

class MutliLayerPerceptrons:
    u"""
    Class for MutliLayerPerceptrons
    """

    def __init__(self, dim_input_signal, dim_hidden, dim_output_signal, rand_obj):

        self.dim_input_signal = dim_input_signal
        self.dim_hidden = dim_hidden
        self.dim_output_signal = dim_output_signal

        self.weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
        self.biases = [0] * dim_output_signal

        if rand_obj is None:
            rand_obj = random()

        self.rand_obj = rand_obj

        # construct hidden layer with tanh as activation function
        hidden_layer = HiddenLayer(dim_input_signal, dim_output_signal, None, None, rand_obj, "Tanh")

        # construct output layer i.e. multi-class logistic layer
        logisticLayer = LogisticRegression(dim_hidden, dim_output_signal)

if __name__ == '__main__':
    CNT_PATTERN = 2
    CNT_TRAIN_DATA = 4 # number of training data
    CNT_TEST_DATA = 4 # number of test data
    DIM_INPUT_SIGNAL = 2 # dimensions of input data
    DIM_HIDDEN = 3 # dimensions of hidden
    DIM_OUTPUT_SIGNAL = CNT_PATTERN # dimensions of output data

    rand_obj = random.Random()
    rand_obj.seed(1234)

    classifier = MutliLayerPerceptrons(DIM_INPUT_SIGNAL, DIM_HIDDEN, DIM_OUTPUT_SIGNAL, rand_obj)


