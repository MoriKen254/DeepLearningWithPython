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

    use_csv = False
    # get argument
    if len(sys.argv) > 1:
        if sys.argv[1] == 'use_csv':
            use_csv = True

    if use_csv:
        file_dir = '../data/DeepNeuralNetworks/RestrictedBoltzmannMachines/'
        for pattern_idx in range(CNT_PATTERN):  # train for each pattern. pattern_idx < 3

            # create training data
            f = open(file_dir  + 'train' + str(pattern_idx + 1) + '.csv', 'r')
            reader = csv.reader(f)
            for n in range(CNT_TRAIN_DATA_EACH): # train for the number of data set for each pattern. n < 200
                train_data_idx = pattern_idx * CNT_TRAIN_DATA_EACH + n
                data = reader.next()
                for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                    train_input_data_set[train_data_idx][visible_idx] = float(data[visible_idx])
            f.close()

            # create test data
            f = open(file_dir  + 'test' + str(pattern_idx + 1) + '.csv', 'r')
            reader = csv.reader(f)
            for n in range(CNT_TEST_DATA_EACH): # train for the number of data set for each pattern. n < 200
                test_data_idx = pattern_idx * CNT_TEST_DATA_EACH + n
                data = reader.next()
                for visible_idx in range(DIM_VISIBLE): # visible_idx < 4
                    test_input_data_set[test_data_idx][visible_idx] = float(data[visible_idx])
            f.close()

    else:
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


    if use_csv:
        print 'Read random data set from csv file.'
        f = open('../data/DeepNeuralNetworks/RestrictedBoltzmannMachines/random_index.csv', 'r')
        reader = csv.reader(f)
        for i in range(CNT_MIN_BATCH):
            for j in range(MIN_BATCH_SIZE):
                idx = int(float(reader.next()[0]))
                train_input_data_set_min_batch[i][j] = train_input_data_set[idx]
        f.close()

    else:
        # create minbatches with training data
        for i in range(CNT_MIN_BATCH):
            for j in range(MIN_BATCH_SIZE):
                idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
                train_input_data_set_min_batch[i][j] = train_input_data_set[idx]

    #
    # Build Multi-Layer Perceptrons model
    #

    # construct
    rbm = RestrictedBoltzmannMachines(DIM_VISIBLE, DIM_HIDDEN, None, None, None, rand_obj, use_csv)

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
