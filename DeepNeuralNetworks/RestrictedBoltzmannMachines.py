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

sys.path.append('../SingleLayerNeuralNetworks')
from LogisticRegression import LogisticRegression

sys.path.append('../MultiLayerNeuralNetworks')
from HiddenLayer import HiddenLayer

sys.path.append('../util')
from RandomGenerator import Uniform, Binomial

class RestrictedBoltzmannMachines:
    u"""
    Class for MutliLayerPerceptrons
    """

    def __init__(self, dim_visible, dim_hidden, weights, hidden_biases, visible_biases, rand_obj, use_csv=False):

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
        self.hidden_bias = hidden_biases_tmp
        self.visible_biases = visible_biases_tmp
        self.rand_obj = rand_obj


    def train(self, input_signals, teacher_labels, min_batch_size, learning_rate):

        hidden_output = [[0] * min_batch_size for i in range(min_batch_size)]
        y_err_arr = [[0] * self.dim_output_signal for i in range(min_batch_size)]

        # forward hidden layer
        for i in range(min_batch_size):
            hidden_output[i] = self.hidden_layer.foward(input_signals[i])

        # forward & backward output layer
        # delta = y - t ... (2.5.32)
        y_err_arr = self.logisticLayer.train(hidden_output, teacher_labels, min_batch_size, learning_rate)

        # backward hidden layer (backpropagate)
        self.hidden_layer.backward(input_signals, hidden_output, y_err_arr, self.logisticLayer.weights,
                                   min_batch_size, learning_rate)

    def predict(self, input_signals):
        # a_j = Sum{ w^T * x + b } ... (2.5.25)
        # z_j = h(a_j) ... (2.5.26)
        hidden_outputs = self.hidden_layer.output(input_signals)
        # a = Sum{ w^T * x + b } where a if from sigma(a) = sigma(w^T * x + b) ... (2.5.9)
        # y_k = exp(a_k) / Sum{ exp(a_k) } = exp(w_k^T * x + b_k) / Sum{ exp(w_k^T * x + b_k) } ... (2.5.18)
        # convert to binary label [0 of 1]
        return self.logisticLayer.predict(hidden_outputs)


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

    # DIM_INPUT_SIGNAL    = 2             # dimensions of input data
    # DIM_OUTPUT_SIGNAL   = CNT_PATTERN   # dimensions of output data

    # input data for training
    train_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TRAIN_DATA)]
    # output data (label) for training
    #train_teacher_labels = [0] * CNT_TRAIN_DATA

    # input data for test
    test_input_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA)]
    # label for inputs
    #test_teacher_labels = [0] * CNT_TEST_DATA
    # output data predicted by the model
    test_restricted_data_set = [[0] * DIM_VISIBLE for j in range(CNT_TEST_DATA)]

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
    classifier = RestrictedBoltzmannMachines(DIM_VISIBLE, DIM_HIDDEN, None, None, None, rand_obj)

    # train
    for epoch in range(EPOCHS):   # training epochs
        for (train_input_data_min_batch, train_teacher_data_min_batch) in \
                zip(train_input_data_set_min_batch, train_teacher_data_set_min_batch):
            classifier.train(train_input_data_min_batch, train_teacher_data_min_batch, MIN_BATCH_SIZE, learning_rate)

        if epoch%10 == 0:
            print 'epoch = %.lf' % epoch

    # test
    for i, test_input_data in enumerate(test_input_data_set):
        test_predict_output_labels[i] = classifier.predict(test_input_data)

    #
    # Evaluate the model
    #
    confusion_matrix = [[0] * CNT_PATTERN for j in range(CNT_PATTERN)]
    accuracy = 0.
    precision = [0] * CNT_PATTERN
    recall = [0] * CNT_PATTERN

    for test_predict_output_label, test_teacher_labels in zip(test_predict_output_labels, test_teacher_labels):
        predicted_idx = test_predict_output_label.index(1)
        actual_idx = test_teacher_labels.index(1)

        confusion_matrix[actual_idx][predicted_idx] += 1

    for i in range(CNT_PATTERN):
        col = 0.
        row = 0.

        for j in range(CNT_PATTERN):
            if i == j:
                accuracy += confusion_matrix[i][j]
                precision[i] += confusion_matrix[j][i]
                recall[i] += confusion_matrix[i][j]

            col += confusion_matrix[j][i]
            row += confusion_matrix[i][j]

        precision[i] /= col
        recall[i] /= row

    accuracy /= CNT_TEST_DATA

    print '--------------------'
    print 'MLP model evaluation'
    print '--------------------'
    print 'Accuracy:  %.1f %%' % (accuracy * 100)
    print 'Precision:'
    for i, precision_elem in enumerate(precision):
        print 'class %d: %.1f %%' % (i+1, precision_elem * 100)
    print 'Recall:'
    for i, recall_elem in enumerate(recall):
        print 'class %d: %.1f %%' % (i+1, recall_elem * 100)


