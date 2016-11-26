#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

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

    def __init__(self, dim_visible, dim_hidden, weights, hidden_biases, visible_biases, rand_obj):

        if rand_obj is None:
            rand_obj = random(1234)

        self.rand_obj = rand_obj

        if weights is None:
            weights_tmp = [[0] * dim_visible for j in range(dim_hidden)]

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
                    gradients_hidden_b_tmp[j] += weight_elem * (input_signal_elem - output_signal)

                gradients_hidden_b_tmp[j] *= hidden * (1 - hidden)
                gradients_hidden_b[j] += gradients_hidden_b_tmp[j]

            # weights
            for j, (gradient_hidden_b_tmp, hidden) in enumerate(zip(gradients_hidden_b_tmp, hiddens)):
                for i, (corrupted_input_signal, gradient_visible_b_tmp) in enumerate(zip(corrupted_input_signals, gradients_visible_b_tmp)):
                    gradients_w[j][i] += gradient_hidden_b_tmp * corrupted_input_signal + gradient_visible_b_tmp * hidden

        # update params
        for j, (gradient_w, gradient_hidden_b) in enumerate(zip(gradients_w, gradients_hidden_b)):
            for i, (gradient_w_elem) in enumerate(gradient_w):
                self.weights[j][i] += learning_rate * gradient_w_elem / min_batch_size

            self.hidden_biases[j] += learning_rate * gradient_hidden_b / min_batch_size

        for i, gradient_visible_b in enumerate(gradients_visible_b):
            self.visible_biases[i] += learning_rate * gradient_visible_b / min_batch_size

    def getCorruptedInput(self, input_signals, corruption_level):
        corrupted_input = [0.0] * len(input_signals)

        # add masking noise
        for i, input_signal in enumerate(input_signals):
            rand_val = self.rand_obj.random()

            if rand_val < corruption_level:
                corrupted_input[i] = 0.0
            else:
                corrupted_input[i] = input_signal

        return corrupted_input

    def getHiddenValues(self, input_signal):

        hiddens = [0] * self.dim_hidden

        for j, (weight, hidden_bias) in enumerate(zip(self.weights, self.hidden_biases)):
            for i, (weight_elem, input_signal_elem) in enumerate(zip(weight, input_signal)):
                hiddens[j] += weight_elem * input_signal_elem

            hiddens[j] += hidden_bias
            hiddens[j] = Sigmoid().compute(hiddens[j])

        return hiddens

    def getReconstructedInput(self, hiddens):

        output_signals = [0] * self.dim_visible

        for i, visible_bias in enumerate(self.visible_biases):
            for j, (weight, hidden) in enumerate(zip(self.weights, hiddens)):
                output_signals[i] += weight[i] * hidden

            output_signals[i] += visible_bias
            output_signals[i] = Sigmoid().compute(output_signals[i])

        return output_signals

    def reconstruct(self, input_signals):
        hiddens = self.getHiddenValues(input_signals)
        output_signals = self.getReconstructedInput(hiddens)

        return output_signals

if __name__ == '__main__':

    CNT_TRAIN_DATA_EACH = 200           # for demo
    CNT_TEST_DATA_EACH  = 2             # for demo
    CNT_VISIBLE_EACH    = 4             # for demo
    PROB_NOISE_TRAIN    = 0.05          # for demo
    PROB_NOISE_TEST     = 0.25          # for demo

    CNT_PATTERN         = 3

    CNT_TRAIN_DATA      = CNT_TRAIN_DATA_EACH * CNT_PATTERN # number of training data
    CNT_TEST_DATA       = CNT_TEST_DATA_EACH * CNT_PATTERN  # number of test data

    DIM_VISIBLE         = CNT_VISIBLE_EACH * CNT_PATTERN    # number of test data
    DIM_HIDDEN          = 6             # dimensions of hidden
    CORRUPTION_LEVEL    = 0.2

    # input data for training
    train_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TRAIN_DATA)]
    # input data for test
    test_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA)]
    # output data predicted by the model
    test_restricted_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA)]
    reconstructed_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA)]

    EPOCHS = 1000           # maximum training epochs
    learning_rate = 0.2     # learning rate

    MIN_BATCH_SIZE = 10     # here, we do on-line training
    CNT_MIN_BATCH = CNT_TRAIN_DATA / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * DIM_VISIBLE for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    train_teacher_data_set_min_batch = [[[0] * DIM_VISIBLE for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH)]
    min_batch_indexes = range(CNT_TRAIN_DATA)
    random.shuffle(min_batch_indexes)   # shuffle data index for SGD

    #
    # Create training data and test data for demo.
    #   Data without noise would be:
    #     class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] ... pattern_idx = 0
    #     class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0] ... pattern_idx = 1
    #     class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] ... pattern_idx = 2
    #                |  |  |  |  |  |  |  |  |  |  |  |
    # (visible_idx:  0  1  2  3  0  1  2  3  0  1  2  3)
    #   and to each data, we add some noise.
    #   For example, one of the data in class 1 could be:
    #     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    #

    rand_obj = random.Random()
    rand_obj.seed(1234)

    binomial_train_true = Binomial(1, 1 - PROB_NOISE_TRAIN)
    binomial_train_false = Binomial(1, PROB_NOISE_TRAIN)

    binomial_test_true = Binomial(1, 1 - PROB_NOISE_TEST)
    binomial_test_false = Binomial(1, PROB_NOISE_TEST)

    for pattern_idx in range(CNT_PATTERN):  # train for each pattern. pattern_idx < 3
        # create training data
        for n in range(CNT_TRAIN_DATA_EACH): # train for the number of data set for each pattern. n < 200
            train_data_idx = pattern_idx * CNT_TRAIN_DATA_EACH + n

            for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                is_pattern_idx_in_curr_part = train_data_idx >= CNT_TRAIN_DATA_EACH * pattern_idx and \
                                              train_data_idx <  CNT_TRAIN_DATA_EACH * (pattern_idx + 1)
                is_visible_idx_in_curr_part = visible_idx    >= CNT_VISIBLE_EACH    * pattern_idx and \
                                              visible_idx    <  CNT_VISIBLE_EACH    * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    train_input_data_set[train_data_idx][visible_idx] = binomial_train_true.compute(rand_obj)
                else:
                    train_input_data_set[train_data_idx][visible_idx] = binomial_train_false.compute(rand_obj)

        # create test data
        for n in range(CNT_TEST_DATA_EACH): # train for the number of data set for each pattern. n < 200
            test_data_idx = pattern_idx * CNT_TEST_DATA_EACH + n

            for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                is_pattern_idx_in_curr_part = test_data_idx >= CNT_TEST_DATA_EACH * pattern_idx and \
                                              test_data_idx <  CNT_TEST_DATA_EACH * (pattern_idx + 1)
                is_visible_idx_in_curr_part = visible_idx   >= CNT_VISIBLE_EACH   * pattern_idx and \
                                              visible_idx   <  CNT_VISIBLE_EACH   * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    test_input_data_set[test_data_idx][visible_idx] = binomial_test_true.compute(rand_obj)
                else:
                    test_input_data_set[test_data_idx][visible_idx] = binomial_test_false.compute(rand_obj)

    # create minbatches with training data
    for i in range(CNT_MIN_BATCH):
        for j in range(MIN_BATCH_SIZE):
            idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
            train_input_data_set_min_batch[i][j] = train_input_data_set[idx]

    #
    # Build Denoising AutoEncoders model
    #

    # construct
    dae = DenoisingAutoencoders(DIM_VISIBLE, DIM_HIDDEN, None, None, None, rand_obj)

    # train
    for epoch in range(EPOCHS):   # training epochs
        for (train_input_data_min_batch, train_teacher_data_min_batch) in \
                zip(train_input_data_set_min_batch, train_teacher_data_set_min_batch):
            dae.train(train_input_data_min_batch, MIN_BATCH_SIZE, learning_rate, CORRUPTION_LEVEL)

        learning_rate *= 0.995

        print 'epoch = %.lf' % epoch
        #if epoch%10 == 0:
        #    print 'epoch = %.lf' % epoch

    # test
    for i, test_input_data in enumerate(test_input_data_set):
        reconstructed_data_set[i] = dae.reconstruct(test_input_data)

    # evvaluation
    print '-----------------------------------'
    print 'DA model reconstruction evaluation'
    print '-----------------------------------'

    for pattern in range(CNT_PATTERN):
        print '\n'
        print 'Class%d' % (pattern + 1)
        for n in range(CNT_TEST_DATA_EACH):
            print_str = ''
            idx = pattern * CNT_TEST_DATA_EACH + n

            print_str +=  '['
            for i in range(DIM_VISIBLE - 1):
                print_str +=  '%d, ' % test_input_data_set[idx][i]
            print_str +=  '%d] -> [' % test_input_data_set[idx][i]

            for i in range(DIM_VISIBLE - 1):
                print_str += '%.5f, ' % reconstructed_data_set[idx][i]
            print_str += '%.5f]' % reconstructed_data_set[idx][i]
            print print_str
