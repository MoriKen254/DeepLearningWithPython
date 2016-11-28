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

class DeepBeliefNets:

    def __init__(self, dim_input_signal, dims_hidden_layers, dim_output_layer, rand_obj):
        if rand_obj is None:
            rand_obj = random(1234)
        self.rand_obj = rand_obj

        self.dim_input_signal = dim_input_signal
        self.dims_hidden_layers = dims_hidden_layers
        self.dim_output_signal = dim_output_layer
        self.cnt_layers = len(dims_hidden_layers)
        self.sigmoid_layers = []#HiddenLayer(self.cnt_layers)
        self.rbm_layers = []

        # construct multi-layer
        dim_prev_layer_input = 0
        for i, dim_hidden_layer in enumerate(self.dims_hidden_layers):
            if i == 0:
                dim_curr_layer_input = dim_input_signal
            else:
                dim_curr_layer_input = dim_prev_layer_input

            # construct hidden layers with sigmoid function
            #   weight matrices and bias vectors will be shared with RBM layers
            self.sigmoid_layers.append(HiddenLayer(dim_curr_layer_input, dim_hidden_layer,
                                                   None, None, rand_obj, 'Sigmoid'))

            # construct RBM layers
            self.rbm_layers.append(RestrictedBoltzmannMachines(dim_curr_layer_input, dim_hidden_layer,
                                                               self.sigmoid_layers[i].weights,
                                                               self.sigmoid_layers[i].biases, None, rand_obj))

            dim_prev_layer_input = dim_hidden_layer

        # logistic regression layer for output
        self.logistic_layer = LogisticRegression(self.dims_hidden_layers[self.cnt_layers-1], self.dim_output_signal)

    def pretrain(self, input_signals_arr, min_batch_size, cnt_min_batch, epochs, learning_rate, cd_k_iter):
        input_signals_tmp = [[0] * self.dim_input_signal for j in range(min_batch_size)]
        for layer in range(self.cnt_layers):
            print 'layer ' + str(layer)
            for epoch in range(epochs):
                print ' epoch ' + str(epoch)
                for input_signals in input_signals_arr:
                    # Set input data for current layer
                    if layer == 0:
                        input_signals_tmp = input_signals
                    else:
                        signals_prev_layer = input_signals_tmp
                        dim_hidden_layer = self.dims_hidden_layers[layer-1]
                        input_signals_tmp = [[0] * dim_hidden_layer for j in range(min_batch_size)]

                        for i, signal_prev_layer in enumerate(signals_prev_layer):
                            input_signals_tmp[i] = self.sigmoid_layers[layer-1].output_binomial(signal_prev_layer, rand_obj)

                    self.rbm_layers[layer].contrastiveDivergence(input_signals_tmp, min_batch_size, learning_rate, cd_k_iter)

    def finetune(self, input_signals_arr, input_teachers, cnt_min_batch, learning_rate):
        layer_inputs = [0] * (self.cnt_layers + 1)
        layer_inputs[0] = input_signals_arr
        hiddens_arr = []

        for layer, dim_hidden_layer in enumerate(self.dims_hidden_layers):
            print 'layer foward' + str(layer)
            inputs_layer = []
            hiddens_arr_tmp = [[0] * dim_hidden_layer for j in range(cnt_min_batch)]

            for n, input_signals in enumerate(input_signals_arr):
                print ' input signal ' + str(n)
                if layer == 0:
                    inputs_layer = input_signals
                else:
                    inputs_layer = hiddens_arr[n]

                hiddens_arr_tmp[n] = self.sigmoid_layers[layer].forward(inputs_layer)

            hiddens_arr = hiddens_arr_tmp
            layer_inputs[layer+1] = copy.deepcopy(hiddens_arr)

        # forward & backward output layer
        grad_output = self.logistic_layer.train(hiddens_arr, input_teachers, cnt_min_batch, learning_rate)

        # backward hidden layers
        grad_hidden = [[0] for j in range(1)]
        for layer in reversed(range(self.cnt_layers)):
            print 'layer backword' + str(layer)
            if layer == self.cnt_layers - 1:
                weights_prev = self.logistic_layer.weights
            else:
                weights_prev = self.sigmoid_layers[layer+1].weights
                grad_output = copy.deepcopy(grad_hidden)

            grad_hidden = self.sigmoid_layers[layer].backward(layer_inputs[layer], layer_inputs[layer+1],
                                                              grad_output, weights_prev,
                                                              cnt_min_batch, learning_rate)

    def predict(self, input_signals):
        hiddens = []

        for layer, sigmoid_layeer in enumerate(self.sigmoid_layers):
            layer_inputs = []

            if layer == 0:
                layer_inputs = input_signals
            else:
                layer_inputs = copy.deepcopy(hiddens)

            hiddens = sigmoid_layeer.forward(layer_inputs)

        return self.logistic_layer.predict(hiddens)


if __name__ == '__main__':

    CNT_TRAIN_DATA_EACH_PTN     = 200           # for demo
    CNT_VALID_DATA_EACH_PTN     = 200           # for demo
    CNT_TEST_DATA_EACH_PTN      = 50            # for demo
    DIM_INPUT_EACH_PTN          = 20            # for demo
    PROB_NOISE_TRAIN            = 0.2           # for demo
    PROB_NOISE_TEST             = 0.25          # for demo

    CNT_PATTERN                 = 3

    CNT_TRAIN_DATA_ALL_PTN      = CNT_TRAIN_DATA_EACH_PTN * CNT_PATTERN      # number of all training data
    CNT_VALID_DATA_ALL_PTN      = CNT_VALID_DATA_EACH_PTN * CNT_PATTERN      # number of all validation data
    CNT_TEST_DATA_ALL_PTN       = CNT_TEST_DATA_EACH_PTN * CNT_PATTERN       # number of all test data

    DIM_INPUT_SIGNAL_ALL_PTN    = DIM_INPUT_EACH_PTN * CNT_PATTERN           # dimension of all input signal
    DIM_OUTPUT_SIGNAL_ALL_PTN   = CNT_PATTERN                                # dimension of all output signal
    DIMS_HIDDEN_LAYERS          = [20, 20]
    CD_K_ITERATION              = 1                                          # CD-k in RBM

    # input data for training
    train_input_data_set = [[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(CNT_TRAIN_DATA_ALL_PTN)]

    # input data for validation
    valid_input_data_set = [[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(CNT_VALID_DATA_ALL_PTN)]
    valid_teacher_labels = [[0] * DIM_OUTPUT_SIGNAL_ALL_PTN for j in range(CNT_VALID_DATA_ALL_PTN)]

    test_input_data_set = [[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(CNT_TEST_DATA_ALL_PTN)]
    test_teacher_labels = [[0] * DIM_OUTPUT_SIGNAL_ALL_PTN for j in range(CNT_TEST_DATA_ALL_PTN)]
    # output data predicted by the model
    test_predict_output_labels = [[0] * DIM_OUTPUT_SIGNAL_ALL_PTN for j in range(CNT_TEST_DATA_ALL_PTN)]

    PRETRAIN_EPOCHS = 50            # maximum pre-training epochs
    PRETRAIN_LEARNING_RATE = 0.2    # learning rate for  pre-training
    FINETUNE_EPOCHS = 50            # maximum fine-tune epochs
    finetune_learning_rate = 0.15   # learning rate for  fine-tune

    MIN_BATCH_SIZE = 50
    CNT_MIN_BATCH_TRAIN = CNT_TRAIN_DATA_ALL_PTN / MIN_BATCH_SIZE
    CNT_MIN_BATCH_VALID = CNT_VALID_DATA_ALL_PTN / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH_TRAIN)]
    valid_input_data_set_min_batch = [[[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH_VALID)]
    valid_teacher_data_set_min_batch = [[[0] * DIM_INPUT_SIGNAL_ALL_PTN for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH_VALID)]
    min_batch_indexes = range(CNT_TRAIN_DATA_ALL_PTN)
    random.shuffle(min_batch_indexes)   # shuffle data index for SGD

    #
    # Create training data and test data for demo.
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

            for input_idx in range(DIM_INPUT_SIGNAL_ALL_PTN): # visible_idx < 4
                is_pattern_idx_in_curr_part = train_data_idx >= CNT_TRAIN_DATA_EACH_PTN * pattern_idx and \
                                              train_data_idx < CNT_TRAIN_DATA_EACH_PTN * (pattern_idx + 1)
                is_visible_idx_in_curr_part = input_idx >= DIM_INPUT_EACH_PTN * pattern_idx and \
                                              input_idx < DIM_INPUT_EACH_PTN * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    train_input_data_set[train_data_idx][input_idx] = binomial_train_true.compute(rand_obj)
                else:
                    train_input_data_set[train_data_idx][input_idx] = binomial_train_false.compute(rand_obj)

        # create validation data
        for n in range(CNT_VALID_DATA_EACH_PTN):
            valid_data_idx = pattern_idx * CNT_VALID_DATA_EACH_PTN + n

            for input_idx in range(DIM_INPUT_SIGNAL_ALL_PTN): # visible_idx < 4
                is_pattern_idx_in_curr_part = train_data_idx >= CNT_VALID_DATA_EACH_PTN * pattern_idx and \
                                              train_data_idx < CNT_VALID_DATA_EACH_PTN * (pattern_idx + 1)
                is_visible_idx_in_curr_part = input_idx >= DIM_INPUT_EACH_PTN * pattern_idx and \
                                              input_idx < DIM_INPUT_EACH_PTN * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    valid_input_data_set[valid_data_idx][input_idx] = binomial_train_true.compute(rand_obj)
                else:
                    valid_input_data_set[valid_data_idx][input_idx] = binomial_train_false.compute(rand_obj)

            for output_idx in range(DIM_OUTPUT_SIGNAL_ALL_PTN):
                if output_idx == pattern_idx:
                    valid_teacher_labels[valid_data_idx][output_idx] = 1
                else:
                    valid_teacher_labels[valid_data_idx][output_idx] = 0

        # create test data
        for n in range(CNT_TEST_DATA_EACH_PTN): # train for the number of data set for each pattern. n < 200
            test_data_idx = pattern_idx * CNT_TEST_DATA_EACH_PTN + n

            for input_idx in range(DIM_INPUT_SIGNAL_ALL_PTN): # visible_idx < 4
                is_pattern_idx_in_curr_part = test_data_idx >= CNT_TEST_DATA_EACH_PTN * pattern_idx and \
                                              test_data_idx < CNT_TEST_DATA_EACH_PTN * (pattern_idx + 1)
                is_visible_idx_in_curr_part = input_idx >= DIM_INPUT_EACH_PTN * pattern_idx and \
                                              input_idx < DIM_INPUT_EACH_PTN * (pattern_idx + 1)
                if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                    test_input_data_set[test_data_idx][input_idx] = binomial_test_true.compute(rand_obj)
                else:
                    test_input_data_set[test_data_idx][input_idx] = binomial_test_false.compute(rand_obj)

            for i in range(DIM_OUTPUT_SIGNAL_ALL_PTN):
                if i == pattern_idx:
                    test_teacher_labels[test_data_idx][i] = 1
                else:
                    test_teacher_labels[test_data_idx][i] = 0

    # create minbatches with training data
    for j in range(MIN_BATCH_SIZE):
        for i in range(CNT_MIN_BATCH_TRAIN):
            idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
            train_input_data_set_min_batch[i][j] = train_input_data_set[idx]
        for i in range(CNT_MIN_BATCH_VALID):
            idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
            valid_input_data_set_min_batch[i][j] = valid_input_data_set[idx]
            valid_teacher_data_set_min_batch[i][j] = valid_teacher_labels[idx]

    #
    # Build Deep Belief Nets model
    #

    # construct DBN
    print 'Building the model...'
    classifier = DeepBeliefNets(DIM_INPUT_SIGNAL_ALL_PTN, DIMS_HIDDEN_LAYERS, DIM_OUTPUT_SIGNAL_ALL_PTN, rand_obj)
    print 'done.'

    # pre-training the model
    print 'Pre-training the model...'
    classifier.pretrain(train_input_data_set_min_batch, MIN_BATCH_SIZE, CNT_MIN_BATCH_TRAIN, PRETRAIN_EPOCHS,
                        PRETRAIN_LEARNING_RATE, CD_K_ITERATION)
    # classifier.
    print 'done.'

    # fine-tuning the model
    print 'Fine-Tuning the model...'
    for epoch in range(FINETUNE_EPOCHS):
#        for valid_input_data_min_batch in enumerate(valid_input_data_set_min_batch):
        for batch in range(CNT_MIN_BATCH_VALID):
            classifier.finetune(valid_input_data_set_min_batch[batch], valid_teacher_data_set_min_batch[batch],
                                MIN_BATCH_SIZE, finetune_learning_rate)
        finetune_learning_rate *= 0.98
    # classifier.
    print 'done.'

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

    for test_predict_output_label, test_teacher_label in zip(test_predict_output_labels, test_teacher_labels):
        predicted_idx = test_predict_output_label.index(1)
        actual_idx = test_teacher_label.index(1)

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

    accuracy /= CNT_TEST_DATA_ALL_PTN

    print '-------------------------------'
    print 'DBN Regression model evaluation'
    print '-------------------------------'
    print 'Accuracy:  %.1f %%' % (accuracy * 100)
    print 'Precision:'
    for i, precision_elem in enumerate(precision):
        print 'class %d: %.1f %%' % (i+1, precision_elem * 100)
    print 'Recall:'
    for i, recall_elem in enumerate(recall):
        print 'class %d: %.1f %%' % (i+1, recall_elem * 100)

