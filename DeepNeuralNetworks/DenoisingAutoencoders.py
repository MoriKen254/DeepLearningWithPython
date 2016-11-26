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
import copy

from RestrictedBoltzmannMachines import RestrictedBoltzmannMachines
sys.path.append('../SingleLayerNeuralNetworks')
from LogisticRegression import LogisticRegression
sys.path.append('../MultiLayerNeuralNetworks')
from HiddenLayer import HiddenLayer
sys.path.append('../util')
from ActivationFunction import Sigmoid
from RandomGenerator import Uniform, Binomial


class DenoisingAutoencoders:
    u"""
    Class for RestrictedBoltzmannMachines
    """

    def __init__(self, dim_visible, dim_hidden, weights, hidden_biases, visible_biases, rand_obj, use_csv=False):
        self.use_csv = use_csv

        if rand_obj is None:
            rand_obj = random(1234)

        self.rand_obj = rand_obj

        if weights is None:
            weights_tmp = [[0] * dim_visible for j in range(dim_hidden)]

            if self.use_csv:
                file_dir = '../data/DeepNeuralNetworks/RestrictedBoltzmannMachines/'
                # create training data
                f = open(file_dir  + 'weights_init.csv', 'r')
                reader = csv.reader(f)
                for j in range(dim_hidden):
                    weight_tmp = reader.next()
                    for i in range(dim_visible):
                        weights_tmp[j][i] = float(weight_tmp[i])

            else:
                w = 1. / dim_visible
                random_generator = Uniform(-w, w)

                for j in range(dim_hidden):
                    for i in range(dim_visible):
                        weights_tmp[j][i] = random_generator.compute(rand_obj)


        else:
            weights_tmp = weights

        if hidden_biases is None:
            hidden_biases_tmp = [0] * dim_hidden
        else:
            hidden_biases_tmp = hidden_biases

        if visible_biases is None:
            visible_biases_tmp = [0] * dim_visible
        else:
            visible_biases_tmp = visible_biases

        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.weights = weights_tmp
        self.hidden_biases = hidden_biases_tmp
        self.visible_biases = visible_biases_tmp
        self.rand_obj = rand_obj

    def train(self, input_signals, min_batch_size, learning_rate, corruption_level):

        gradients_w = [[0] * self.dim_visible for i in range(self.dim_hidden)]
        gradients_hidden_b = [0] * self.dim_hidden
        gradients_visible_b = [0] * self.dim_visible

        # forward hidden layer
        for n, input_signal in enumerate(input_signals):

            # hidden parameters for positive phase
            means_prob_hid_pos = [0] * self.dim_hidden   # mean of p( h_j | v^(k) ) for positive phase. 0.0 ~ 1.0
            samples_hid_pos = [0] * self.dim_hidden      # h_j^(k) ~ p( h_j | v^(k) ) for positive phase. 0 or 1

            # visible parameters for negative phase
            means_prob_vis_neg = [0] * self.dim_visible  # mean of p( v_i | h^(k) ) for negative phase. 0.0 ~ 1.0
            samples_vis_neg = [0] * self.dim_visible     # v_i^(k+1) # of p( v_i | h^(k) ) for negative phase. 0 or 1

            # hidden parameters for negative phase
            means_prob_hid_neg = [0] * self.dim_hidden   # mean of p( h_j | v^(k+1) ) for positive phase. 0.0 ~ 1.0
            samples_hid_neg = [0] * self.dim_hidden      # h_j^(k+1) ~ p( h_j | v^(k+1) ) for positive phase. 0 or 1

            # add noise to original inputs
            corrupted_input_signals = self.getCorruptedInput(input_signal, corruption_level)

            # encode
            hiddens = self.getHiddenValues(corrupted_input_signals)

            # decode
            output_signals = self.getReconstructedInput(hiddens)

            # calculate gradients

            # visible biases
            gradients_visible_b_tmp = [0] * self.dim_visible
            for i, (input_elem, output_signal) in enumerate(zip(input_signal, output_signals)):
                gradients_visible_b_tmp[i] = input_elem - output_signal
                gradients_visible_b[i] += gradients_visible_b_tmp[i]

            # hidden biases
            gradients_hidden_b_tmp = [0] * self.dim_hidden
            for j, (weight, hidden) in enumerate(zip(self.weights, hiddens)):
                for i, (input_signal_elem, weight_elem, output_signal) in enumerate(zip(input_signal, weight, output_signals)):
                    gradients_hidden_b_tmp[j][i] += weight_elem * (input_signal_elem - output_signal)

                gradients_hidden_b_tmp[j] *= hidden * (1 - hidden)
                gradients_hidden_b[j] += gradients_hidden_b_tmp[j]

            # weights
            for j, (gradient_hidden_b_tmp, hidden) in enumerate(zip(gradients_hidden_b_tmp, hiddens)):
                for i, (corrupted_input_signal, gradient_visible_b_tmp) in enumerate(zip(corrupted_input_signals, gradients_visible_b_tmp)):
                    gradients_w[j][i] += gradient_hidden_b_tmp * corrupted_input_signal + gradient_visible_b_tmp * hidden

        ####### TODO
        # update params
        for j, (gradient_w, gradient_hidden_b) in enumerate(zip(gradients_w, gradients_hidden_b)):
            for i, (gradient_w_elem) in enumerate(gradient_w):
                self.weights[j][i] += learning_rate * gradient_w_elem / min_batch_size

            self.hidden_biases[j] += learning_rate * gradient_hidden_b / min_batch_size

        for i, gradient_visible_b in enumerate(gradients_visible_b):
            self.visible_biases[i] += learning_rate * gradient_visible_b / min_batch_size

    def getCorruptedInput(self, input_signals, corruption_level):
        corrupted_input = [0] * len(input_signals)

        # add masking noise
        for i, input_signal in enumerate(input_signals):
            rand_val = self.rand_obj.random()

            if rand_val < corruption_level:
                corrupted_input = 0.
            else:
                corrupted_input = input_signal

        return corruption_level

    def getHiddenValues(self, input_signal):

        hiddens = [0] * self.dim_hidden

        for j, weight in enumerate(self.weights):
            for i, (weight_elem, input_signal_elem) in enumerate(zip(weight, input_signal)):
                hiddens[j] += weight_elem * input_signal_elem

            hiddens[j] += self.hidden_biases
            hiddens[j] = Sigmoid().compute(hiddens[j])

        return hiddens

    def getReconstructedInput(self, hiddens):

        output_signals = [0] * self.dim_visible

        for j, weight in enumerate(self.weights):
            for i, (weight_elem, hidden) in enumerate(zip(weight, hiddens)):
                output_signals[j] += weight_elem * hidden

            output_signals[j] += self.visible_biases
            output_signals[j] = Sigmoid().compute(output_signals[j])

        return output_signals

