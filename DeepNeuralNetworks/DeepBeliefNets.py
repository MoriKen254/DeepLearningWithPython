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

class DeepBeliefNets:

    def __init__(self, dim_input_signal, dims_hidden_layers, dim_output_layer, rand_obj, use_csv=False):
        self.use_csv = use_csv

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
        for layer in range(self.cnt_layers):
            for epoch in range(epochs):
                print 'epoch ' + str(epoch)
                for input_signals in input_signals_arr:

                    input_signals_tmp = [[0] * self.dim_input_signal for j in range(min_batch_size)]

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
        
        for layer, dim_hidden_layer in enumerate(self.dims_hidden_layers):

            inputs_layer = []
            hiddens_arr_tmp = [[0] * dim_hidden_layer for j in range(cnt_min_batch)]

            for n, (input_signals, hiddens) in enumerate(zip(input_signals_arr, hiddens_arr_tmp)):
                if layer == 0:
                    inputs_layer = input_signals
                else:
                    inputs_layer = hiddens

                hiddens_arr_tmp[n] = self.sigmoid_layers[layer].forward(inputs_layer)

            hiddens_arr = hiddens_arr_tmp
            #layer_inputs.append(copy.deepcopy(hiddens_arr))
            layer_inputs[layer+1] = copy.deepcopy(hiddens_arr)

        # forward & backward output layer
        grad_output = self.logistic_layer.train(hiddens_arr, input_teachers, cnt_min_batch, learning_rate)

        # backward hidden layers
        grad_hidden = [[0] for j in range(1)]
        for layer in reversed(range(self.cnt_layers)):

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

    CNT_TRAIN_DATA_EACH = 200           # for demo
    CNT_VALID_DATA_EACH = 200      # for demo
    CNT_TEST_DATA_EACH  = 2             # for demo
    CNT_INPUT_EACH      = 20            # for demo
    PROB_NOISE_TRAIN    = 0.2           # for demo
    PROB_NOISE_TEST     = 0.25          # for demo

    CNT_PATTERN         = 3

    CNT_TRAIN_DATA      = CNT_TRAIN_DATA_EACH * CNT_PATTERN       # number of training data
    CNT_VALID_DATA      = CNT_VALID_DATA_EACH * CNT_PATTERN       # number of validation data
    CNT_TEST_DATA       = CNT_TEST_DATA_EACH * CNT_PATTERN        # number of test data

    CNT_INPUT_DATA      = CNT_INPUT_EACH * CNT_PATTERN            # number of input data
    CNT_OUTPUT_DATA     = CNT_PATTERN                             # number of output data
    DIMS_HIDDEN_LAYERS  = [20, 20]
    CD_K_ITERATION      = 1             # CD-k in RBM

    # input data for training
    train_input_data_set = [[0] * CNT_INPUT_DATA for j in range(CNT_TRAIN_DATA)]

    # input data for validation
    valid_input_data_set = [[0] * CNT_INPUT_DATA for j in range(CNT_VALID_DATA)]
    valid_teacher_data_set = [[0] * CNT_OUTPUT_DATA for j in range(CNT_VALID_DATA)]

    test_input_data_set = [[0] * CNT_INPUT_DATA for j in range(CNT_TEST_DATA)]
    test_teacher_data_set = [[0] * CNT_OUTPUT_DATA for j in range(CNT_TEST_DATA)]
    # output data predicted by the model
    predicted_teacher_data_set = [[0] * CNT_OUTPUT_DATA for j in range(CNT_TEST_DATA)]

    PRETRAIN_EPOCHS = 2#00          # maximum pre-training epochs
    PRETRAIN_LEARNING_RATE = 0.2    # learning rate for  pre-training
    FINETUNE_EPOCHS = 2#1000          # maximum fine-tune epochs
    FINETUNE_LEARNING_RATE = 0.15   # learning rate for  fine-tune

    MIN_BATCH_SIZE = 50
    CNT_MIN_BATCH_TRAIN = CNT_TRAIN_DATA / MIN_BATCH_SIZE
    CNT_MIN_BATCH_VALID = CNT_VALID_DATA / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * CNT_INPUT_DATA for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH_TRAIN)]
    valid_input_data_set_min_batch = [[[0] * CNT_INPUT_DATA for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH_VALID)]
    valid_teacher_data_set_min_batch = [[[0] * CNT_INPUT_DATA for j in range(MIN_BATCH_SIZE)]
                                        for k in range(CNT_MIN_BATCH_VALID)]
    min_batch_indexes = range(CNT_TRAIN_DATA)
    random.shuffle(min_batch_indexes)   # shuffle data index for SGD

    #
    # Create training data and test data for demo.
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
                for input_idx in range(DIM_VISIBLE): # visible_idx < 4
                    train_input_data_set[train_data_idx][input_idx] = float(data[input_idx])
            f.close()

            # create test data
            f = open(file_dir  + 'test' + str(pattern_idx + 1) + '.csv', 'r')
            reader = csv.reader(f)
            for n in range(CNT_TEST_DATA_EACH): # train for the number of data set for each pattern. n < 200
                test_data_idx = pattern_idx * CNT_TEST_DATA_EACH + n
                data = reader.next()
                for input_idx in range(DIM_VISIBLE): # visible_idx < 4
                    test_input_data_set[test_data_idx][input_idx] = float(data[input_idx])
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

                for input_idx in range(CNT_INPUT_DATA): # visible_idx < 4
                    is_pattern_idx_in_curr_part = train_data_idx >= CNT_TRAIN_DATA_EACH * pattern_idx and \
                                                  train_data_idx <  CNT_TRAIN_DATA_EACH * (pattern_idx + 1)
                    is_visible_idx_in_curr_part = input_idx >= CNT_INPUT_EACH * pattern_idx and \
                                                  input_idx < CNT_INPUT_EACH * (pattern_idx + 1)
                    if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                        train_input_data_set[train_data_idx][input_idx] = binomial_train_true.compute(rand_obj)
                    else:
                        train_input_data_set[train_data_idx][input_idx] = binomial_train_false.compute(rand_obj)

            # create validation data
            for n in range(CNT_VALID_DATA_EACH):
                valid_data_idx = pattern_idx * CNT_VALID_DATA_EACH + n

                for input_idx in range(CNT_INPUT_DATA): # visible_idx < 4
                    is_pattern_idx_in_curr_part = train_data_idx >= CNT_VALID_DATA_EACH * pattern_idx and \
                                                  train_data_idx <  CNT_VALID_DATA_EACH * (pattern_idx + 1)
                    is_visible_idx_in_curr_part = input_idx >= CNT_INPUT_EACH * pattern_idx and \
                                                  input_idx <  CNT_INPUT_EACH * (pattern_idx + 1)
                    if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                        valid_input_data_set[valid_data_idx][input_idx] = binomial_train_true.compute(rand_obj)
                    else:
                        valid_input_data_set[valid_data_idx][input_idx] = binomial_train_false.compute(rand_obj)

                for output_idx in range(CNT_OUTPUT_DATA):
                    if output_idx == pattern_idx:
                        valid_teacher_data_set[valid_data_idx][output_idx] = 1
                    else:
                        valid_teacher_data_set[valid_data_idx][output_idx] = 0

            # create test data
            for n in range(CNT_TEST_DATA_EACH): # train for the number of data set for each pattern. n < 200
                test_data_idx = pattern_idx * CNT_TEST_DATA_EACH + n

                for input_idx in range(CNT_INPUT_DATA): # visible_idx < 4
                    is_pattern_idx_in_curr_part = test_data_idx >= CNT_TEST_DATA_EACH * pattern_idx and \
                                                  test_data_idx <  CNT_TEST_DATA_EACH * (pattern_idx + 1)
                    is_visible_idx_in_curr_part = input_idx >= CNT_INPUT_EACH * pattern_idx and \
                                                  input_idx <  CNT_INPUT_EACH * (pattern_idx + 1)
                    if is_pattern_idx_in_curr_part and is_visible_idx_in_curr_part:
                        test_input_data_set[test_data_idx][input_idx] = binomial_test_true.compute(rand_obj)
                    else:
                        test_input_data_set[test_data_idx][input_idx] = binomial_test_false.compute(rand_obj)


    if use_csv:
        print 'Read random data set from csv file.'
        f = open('../data/DeepNeuralNetworks/RestrictedBoltzmannMachines/random_index.csv', 'r')
        reader = csv.reader(f)
        for j in range(MIN_BATCH_SIZE):
            for i in range(CNT_MIN_BATCH_TRAIN):
                idx = int(float(reader.next()[0]))
                train_input_data_set_min_batch[i][j] = train_input_data_set[idx]
        f.close()

    else:
        # create minbatches with training data
        for j in range(MIN_BATCH_SIZE):
            for i in range(CNT_MIN_BATCH_TRAIN):
                idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
                train_input_data_set_min_batch[i][j] = train_input_data_set[idx]
            for i in range(CNT_MIN_BATCH_VALID):
                idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
                valid_input_data_set_min_batch[i][j] = valid_input_data_set[idx]
                valid_teacher_data_set_min_batch[i][j] = valid_teacher_data_set[idx]

    #
    # Build Deep Belief Nets model
    #

    # construct DBN
    print 'Building the model...'
    classifier = DeepBeliefNets(CNT_INPUT_DATA, DIMS_HIDDEN_LAYERS, CNT_OUTPUT_DATA, rand_obj)
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
                                CNT_MIN_BATCH_VALID, FINETUNE_LEARNING_RATE)
        FINETUNE_LEARNING_RATE *= 0.98
    # classifier.
    print 'done.'

    # test
    for i, test_input_data in enumerate(test_input_data_set):
        predicted_teacher_data_set[i] = classifier.predict(test_input_data)

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
