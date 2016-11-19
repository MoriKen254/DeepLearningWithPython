#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import csv
import random
import sys

sys.path.append('../util')
from RandomGenerator import Uniform, Binomial
from ActivationFunction import Sigmoid, Tanh, ReLU

class HiddenLayer:

    def __init__(self, dim_input_signal, dim_output_signal, weights, biases, rand_obj, activation, use_csv=False):

        if rand_obj is None:
            rand_obj = random(1234)

        if weights is None:
            weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
            w = 1./dim_input_signal

            if use_csv:
                f = open('../data/MultiLayerPerceptrons/weights_hidden.csv', 'r')
                reader = csv.reader(f)
                for j, row in enumerate(reader):
                    for i in range(dim_input_signal):
                        weights[j][i] = float(row[i])
            else:
                for j in range(dim_output_signal):
                    for i in range(dim_input_signal):
                        random_generator = Uniform(-w, w)
                        weights[j][i] = random_generator.compute(rand_obj)

        if biases is None:
            biases = [0] * dim_output_signal

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
        after_activations = [0] * self.dim_output_signal

        # a_j = Sum{ w^T * x + b } ... (2.5.25)
        for j, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = 0
            for w, input_signal in zip(weight, input_signals):
                pre_activation += w * input_signal

            pre_activation += bias # linear output

            # z_j = h(a_j) ... (2.5.26)
            after_activations[j] = self.activation.compute(pre_activation)

        return after_activations

    def foward(self, input_signals):
        return self.output(input_signals)

    def backward(self, input_signals, hidden_outputs, y_err_arr, weights_prev, min_batch_size, learning_rate):

        back_propagation_err = [[0] * self.dim_output_signal for i in range(min_batch_size)]

        gradients_w = [[0] * self.dim_input_signal for i in range(self.dim_output_signal)]
        gradients_b = [0] * self.dim_output_signal

        # train with SGD
        # calculate backpropagation error to get gradient of w, b
        for n, (input_signal, hidden_output, y_err) in  enumerate(zip(input_signals, hidden_outputs, y_err_arr)): # n < minbatchsize

            for j in range(self.dim_output_signal): # j < dim_output_signal of current layer

                # delta_j = h'(a_j) * Sum { w_kj * delta_k } ... (2.5.33)
                for k, (y_err_elem, weight_prev) in enumerate(zip(y_err, weights_prev)): # k < dim_output_signal of previous layer
                    # Sum { w_kj * delta_k }
                    back_propagation_err[n][j] += weight_prev[j] * y_err_elem

                # h'(a_j)
                back_propagation_err[n][j] *= self.activation.differentiate(hidden_output[j])

                # dE_n/dw_ji = sigma_j * x_i ... (2.5.14), (2.5.30)
                for i, input_signal_elem in enumerate(input_signal): # i < dim_input_signal of current layer
                    gradients_w[j][i] += back_propagation_err[n][j] * input_signal_elem

                # dE_n/db_j = sigma_j ... (2.5.15), (2.5.31)
                gradients_b[j] += back_propagation_err[n][j]

        # update params
        for j, (gradient_w, gradient_b) in enumerate(zip(gradients_w, gradients_b)):
            # w^(k+1) = w^(k) - eta * dE/dw ... (2.5.16)
            for i, gradient_w_elem in enumerate(gradient_w):
                self.weights[j][i] -= learning_rate * gradient_w_elem / min_batch_size

            # b^(k+1) = b^(k) - eta * dE/db ... (2.5.17)
            self.biases[j] -= learning_rate * gradient_b / min_batch_size

        return back_propagation_err

    def output_binomial(self, input_signals, rand_obj):
        output_val = [0] * self.dim_output_signal

        input_casts = [0] * len(input_signals)
        for i, input_elem in enumerate(input_signals):
            input_casts[i] = float(input_elem)

        out = self.output(input_casts)

        for j, out_elem in enumerate(out):
            binomial = Binomial(1, out_elem)
            output_val[j] = binomial.compute(rand_obj)

        return output_val

    def forward(self, input_signals):
        return self.output(input_signals)
