#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""


import sys
import random

sys.path.append('../SingleLayerNeuralNetworks')
from LogisticRegression import LogisticRegression

sys.path.append('../MultiLayerNeuralNetworks')
from HiddenLayer import HiddenLayer

sys.path.append('../util')
from ActivationFunction import Sigmoid
from RandomGenerator import Uniform, Binomial

class RestrictedBoltzmannMachines:
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


    def contrastiveDivergence(self, input_signals, min_batch_size, learning_rate, cd_k_iteration):

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

            # CD-k: CD-1 is enough for sampling (i.e. k=1)
            # h^(0) ~ p(h_j|v^(0))
            self.sampleHidGivenVis(input_signal, means_prob_hid_pos, samples_hid_pos)

            for step in range(cd_k_iteration):

                # Gibbs sampling
                if step == 0:
                    self.gibbsHidVisHid(samples_hid_pos, means_prob_vis_neg,
                                        samples_vis_neg, means_prob_hid_neg, samples_hid_neg)
                else:
                    self.gibbsHidVisHid(samples_hid_neg, means_prob_vis_neg,
                                        samples_vis_neg, means_prob_hid_neg, samples_hid_neg)

            # calculate gradients
            for j, (mean_prob_hid_pos, mean_prob_hid_neg) in enumerate(zip(means_prob_hid_pos, means_prob_hid_neg)):
                for i, (input_elem, sample_vis_neg) in enumerate(zip(input_signal, samples_vis_neg)):
                    gradients_w[j][i] += mean_prob_hid_pos * input_elem - mean_prob_hid_neg * sample_vis_neg

                gradients_hidden_b[j] += mean_prob_hid_pos - mean_prob_hid_neg

            for i, (input_elem, sample_vis_neg) in enumerate(zip(input_signal, samples_vis_neg)):
                gradients_visible_b[i] += input_elem - sample_vis_neg

        # update params
        for j, (gradient_w, gradient_hidden_b) in enumerate(zip(gradients_w, gradients_hidden_b)):
            for i, (gradient_w_elem) in enumerate(gradient_w):
                self.weights[j][i] += learning_rate * gradient_w_elem / min_batch_size

            self.hidden_biases[j] += learning_rate * gradient_hidden_b / min_batch_size

        for i, gradient_visible_b in enumerate(gradients_visible_b):
            self.visible_biases[i] += learning_rate * gradient_visible_b / min_batch_size


    def gibbsHidVisHid(self, samples_hid_init, means_prob_vis_neg, samples_vis_neg,
                                               means_prob_hid_neg, samples_hid_neg):

        self.sampleVisGivenHid(samples_hid_init, means_prob_vis_neg, samples_vis_neg) # h_j^(k)   ~ p(h_j|v^(k))
        self.sampleHidGivenVis(samples_vis_neg,  means_prob_hid_neg, samples_hid_neg) # v_i^(k+1) ~ p(v_i|h^(k))

    def sampleHidGivenVis(self, samples_vis, mean_prob_vis, samples_hid_output):

        for j, (weight, hid_bias) in enumerate(zip(self.weights, self.hidden_biases)):
            mean_prob_vis[j] = self.propup(samples_vis, weight, hid_bias)  # v^(k) of p(h_j|v^(k))
            rand_gen = Binomial(1, mean_prob_vis[j])
            samples_hid_output[j] = rand_gen.compute(self.rand_obj) # h_j^(k) ~ p(h_j|v^(k))

    def sampleVisGivenHid(self, samples_hid, mean_prob_hid, samples_vis_output):

        for i, vis_bias in enumerate(self.visible_biases):
            mean_prob_hid[i] = self.propdown(samples_hid, i, vis_bias)  # h^(k) of p(v_i|h^(k))
            rand_gen = Binomial(1, mean_prob_hid[i])
            samples_vis_output[i] = rand_gen.compute(self.rand_obj) # v_i^(k+1) ~ p(v_i|h^(k))

    def propup(self, samples_visible, weight, hidden_bias):

        pre_activation = 0.

        for weight_elem, sample_visible in zip(weight, samples_visible):
            pre_activation += weight_elem * sample_visible

        pre_activation += hidden_bias

        return Sigmoid().compute(pre_activation)

    def propdown(self, samples_hidden, idx_hid, visible_bias):

        pre_activation = 0.

        for j, sample_hidden in enumerate(samples_hidden):
            pre_activation += self.weights[j][idx_hid] * sample_hidden

        pre_activation += visible_bias

        return Sigmoid().compute(pre_activation)

    def reconstruct(self, input_signals_vis):

        reconstructs_vis = [0] * self.dim_visible
        means_prob_hid = [0] * self.dim_hidden

        for j, (weight, hid_bias) in enumerate(zip(self.weights, self.hidden_biases)):
            means_prob_hid[j] = self.propup(input_signals_vis, weight, hid_bias)

        for i, visible_bias in enumerate(self.visible_biases):
            pre_activation = 0.0

            for j, (weight, mean_prob_hid) in enumerate(zip(self.weights, means_prob_hid)):
                pre_activation += weight[i] * mean_prob_hid

            pre_activation += visible_bias
            reconstructs_vis[i] = Sigmoid().compute(pre_activation)

        return reconstructs_vis


if __name__ == '__main__':

    CNT_TRAIN_DATA_EACH_PTN     = 200           # for demo
    CNT_TEST_DATA_EACH_PTN      = 2             # for demo
    CNT_VISIBLE_EACH_PTN        = 4             # for demo
    PROB_NOISE_TRAIN            = 0.05          # for demo
    PROB_NOISE_TEST             = 0.25          # for demo

    CNT_PATTERN                 = 3

    CNT_TRAIN_DATA_ALL_PTN      = CNT_TRAIN_DATA_EACH_PTN * CNT_PATTERN     # number of training data
    CNT_TEST_DATA_ALL_PTN       = CNT_TEST_DATA_EACH_PTN * CNT_PATTERN      # number of test data

    DIM_VISIBLE                 = CNT_VISIBLE_EACH_PTN * CNT_PATTERN        # number of test data
    DIM_HIDDEN                  = 6                                         # dimensions of hidden

    # input data for training
    train_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TRAIN_DATA_ALL_PTN)]
    # input data for test
    test_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA_ALL_PTN)]
    # output data predicted by the model
    test_restricted_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA_ALL_PTN)]
    reconstructed_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA_ALL_PTN)]

    EPOCHS = 1000           # maximum training epochs
    learning_rate = 0.2     # learning rate

    MIN_BATCH_SIZE = 10     # here, we do on-line training
    CNT_MIN_BATCH = CNT_TRAIN_DATA_ALL_PTN / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * DIM_VISIBLE for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    train_teacher_data_set_min_batch = [[[0] * DIM_VISIBLE for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH)]
    min_batch_indexes = range(CNT_TRAIN_DATA_ALL_PTN)
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
        for n in range(CNT_TRAIN_DATA_EACH_PTN): # train for the number of data set for each pattern. n < 200
            train_data_idx = pattern_idx * CNT_TRAIN_DATA_EACH_PTN + n

            for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                is_pattern_idx_in_curr_part = train_data_idx >= CNT_TRAIN_DATA_EACH_PTN * pattern_idx and \
                                              train_data_idx < CNT_TRAIN_DATA_EACH_PTN * (pattern_idx + 1)
                is_visible_idx_in_curr_part = visible_idx >= CNT_VISIBLE_EACH_PTN * pattern_idx and \
                                              visible_idx < CNT_VISIBLE_EACH_PTN * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    train_input_data_set[train_data_idx][visible_idx] = binomial_train_true.compute(rand_obj)
                else:
                    train_input_data_set[train_data_idx][visible_idx] = binomial_train_false.compute(rand_obj)

        # create test data
        for n in range(CNT_TEST_DATA_EACH_PTN): # train for the number of data set for each pattern. n < 200
            test_data_idx = pattern_idx * CNT_TEST_DATA_EACH_PTN + n

            for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                is_pattern_idx_in_curr_part = test_data_idx >= CNT_TEST_DATA_EACH_PTN * pattern_idx and \
                                              test_data_idx < CNT_TEST_DATA_EACH_PTN * (pattern_idx + 1)
                is_visible_idx_in_curr_part = visible_idx >= CNT_VISIBLE_EACH_PTN * pattern_idx and \
                                              visible_idx < CNT_VISIBLE_EACH_PTN * (pattern_idx + 1)
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
    # Build Multi-Layer Perceptrons model
    #

    # construct
    rbm = RestrictedBoltzmannMachines(DIM_VISIBLE, DIM_HIDDEN, None, None, None, rand_obj)

    # train
    for epoch in range(EPOCHS):   # training epochs
        for (train_input_data_min_batch, train_teacher_data_min_batch) in \
                zip(train_input_data_set_min_batch, train_teacher_data_set_min_batch):
            rbm.contrastiveDivergence(train_input_data_min_batch, MIN_BATCH_SIZE, learning_rate, 1)

        learning_rate *= 0.995

        print 'epoch = %.lf' % epoch
        #if epoch%10 == 0:
        #    print 'epoch = %.lf' % epoch

    # test
    for i, test_input_data in enumerate(test_input_data_set):
        reconstructed_data_set[i] = rbm.reconstruct(test_input_data)

    # evvaluation
    print '-----------------------------------'
    print 'RBM model reconstruction evaluation'
    print '-----------------------------------'

    for pattern in range(CNT_PATTERN):
        print '\n'
        print 'Class%d' % (pattern + 1)
        for n in range(CNT_TEST_DATA_EACH_PTN):
            print_str = ''
            idx = pattern * CNT_TEST_DATA_EACH_PTN + n

            print_str +=  '['
            for i in range(DIM_VISIBLE - 1):
                print_str +=  '%d, ' % test_input_data_set[idx][i]
            print_str +=  '%d] -> [' % test_input_data_set[idx][i]

            for i in range(DIM_VISIBLE - 1):
                print_str += '%.5f, ' % reconstructed_data_set[idx][i]
            print_str += '%.5f]' % reconstructed_data_set[idx][i]
            print print_str

