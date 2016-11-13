#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import sys
import random

sys.path.append('../util')
from ActivationFunction import Softmax
from GaussianDistribution import GaussianDistribution

class LogisticRegression:
    u"""
    Class for LogisticRegression
    """

    def __init__(self, dim_input_signal, dim_output_signal):

        self.dim_input_signal = dim_input_signal
        self.dim_output_signal = dim_output_signal

        self.weights = [[0] * dim_input_signal for i in range(dim_output_signal)]
        self.biases = [0] * dim_output_signal


    def train(self, input_signals, teacher_labels, min_batch_size, learning_rate):

        gradients_w = [[0] * self.dim_input_signal for i in range(self.dim_output_signal)]
        gradients_b = [0] * self.dim_output_signal

        y_arr_batch = [[0] * self.dim_output_signal for i in range(min_batch_size)]

        # train with SGD
        # 1. calculate gradient of gradients_w, gradients_b
        for i, (input_signal, teacher_label, y_batch) in zip(enumerate(input_signals), teacher_labels, y_arr_batch):
            predicted_y_arr = output(input_signal)


    def output(self, input_signals):
        pre_activations = [0] * self.dim_output_signal

        for j, (weight, bias) in zip(enumerate(self.weights), self.biases):
            for i, (w, input_signal) in zip(enumerate(self.weights), input_signals):
                pre_activations[j] += w * input_signal

            pre_activations[j] += bias

        Sofmax.compute(pre_activations, self.dim_output_signal)


    def predict(self, input_signals):

        pre_activation = 0.0

        # check array size
        if len(input_signals) is not len(self.weights):
            return

        for (weight, input_signal) in zip(self.weights, input_signals):
            pre_activation += weight * input_signal

        activation_function = Step()
        return activation_function.compute(pre_activation)


if __name__ == '__main__':
    CNT_PATTERN = 3
    CNT_TRAIN_DATA = 400 * CNT_PATTERN # number of training data
    CNT_TEST_DATA = 60 * CNT_PATTERN # number of test data
    DIM_INPUT_SIGNAL = 2 # dimensions of input data
    DIM_OUTPUT_SIGNAL = CNT_PATTERN # dimensions of output data

    # input data for training
    train_input_data_set = [[0] * DIM_INPUT_SIGNAL for j in range(CNT_TRAIN_DATA)]
    # output data (label) for training
    train_teacher_labels = [0] * CNT_TRAIN_DATA

    # input data for test
    test_input_data_set = [[0] * DIM_INPUT_SIGNAL for j in range(CNT_TEST_DATA)]
    # label for inputs
    test_teacher_labels = [0] * CNT_TEST_DATA
    # output data predicted by the model
    test_predict_output_labels = [0] * CNT_TEST_DATA

    EPOCHS = 1000   # maximum training epochs
    LEARNING_RATE = 0.2 # learning rate

    MIN_BATCH_SIZE = 50 # number of data in each minbatch
    CNT_MIN_BATCH = CNT_TRAIN_DATA / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * DIM_INPUT_SIGNAL for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    train_teacher_data_set_min_batch = [[[0] * DIM_OUTPUT_SIGNAL for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    min_batch_indexes = range(CNT_TRAIN_DATA)
    random.shuffle(min_batch_indexes)

    #
    # Create training data and test data for demo.
    #
    # Let training data set for each class follow Normal (Gaussian) distribution here:
    #   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
    #   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
    #

    rand_obj = random.Random()
    rand_obj.seed(1234)

    gaussian1 = GaussianDistribution(-2.0, 1.0, rand_obj)
    gaussian2 = GaussianDistribution(2.0, 1.0, rand_obj)
    gaussian3 = GaussianDistribution(0.0, 1.0, rand_obj)

    # # data set in class 1
    # for i in range(0, CNT_TRAIN_DATA/2):
    #     train_input_data_set[i][0] = gaussian1.get_random()
    #     train_input_data_set[i][1] = gaussian2.get_random()
    #     train_teacher_labels[i] = 1
    # for i in range(0, CNT_TEST_DATA/2):
    #     test_input_data_set[i][0] = gaussian1.get_random()
    #     test_input_data_set[i][1] = gaussian2.get_random()
    #     test_teacher_labels[i] = 1

    # # data set in class 2
    # for i in range(CNT_TRAIN_DATA/2, CNT_TRAIN_DATA):
    #     train_input_data_set[i][0] = gaussian2.get_random()
    #     train_input_data_set[i][1] = gaussian1.get_random()
    #     train_teacher_labels[i] = -1
    # for i in range(CNT_TEST_DATA/2, CNT_TEST_DATA):
    #     test_input_data_set[i][0] = gaussian2.get_random()
    #     test_input_data_set[i][1] = gaussian1.get_random()
    #     test_teacher_labels[i] = -1

    # #
    # # build Perceptron model
    # #
    # epoch = 0   # training epochs

    # # construct perceptrons
    # classifier = Perceptrons(DIM_INPUT_SIGNAL)

    # # train models
    # classified_sum_prev = 0
    # while True:
    #     classified_sum = 0

    #     for (train_input_data, train_teacher_label) in zip(train_input_data_set, train_teacher_labels):
    #         classified_sum += classifier.train(train_input_data, train_teacher_label, LEARNING_RATE)

    #     if classified_sum == CNT_TRAIN_DATA: # if all classes are correctly classified
    #         break

    #     if abs(classified_sum - classified_sum_prev) == 0: # if converged
    #         break
    #     classified_sum_prev = classified_sum

    #     epoch += 1
    #     if (epoch > EPOCHS): # if not converged after enough trials
    #         break

    # # test
    # for i, test_input_data in enumerate(test_input_data_set):
    #     test_predict_output_labels[i] = classifier.predict(test_input_data)

    # #
    # # Evaluate the model
    # #
    # confusion_matrix = [[0] * 2 for j in range(0, 2)]
    # accuracy = 0.
    # precision = 0.
    # recall = 0.

    # for (test_predict_output_label, test_teacher_label) in zip(test_predict_output_labels, test_teacher_labels):
    #     if test_predict_output_label > 0:
    #         if(test_teacher_label > 0):
    #             accuracy += 1
    #             precision += 1
    #             recall += 1
    #             confusion_matrix[0][0] += 1
    #         else:
    #             confusion_matrix[1][0] += 1
    #     else:
    #         if(test_teacher_label > 0):
    #             confusion_matrix[0][1] += 1
    #         else:
    #             accuracy += 1
    #             confusion_matrix[1][1] += 1

    # accuracy /= CNT_TEST_DATA
    # if confusion_matrix[0][0] + confusion_matrix[1][0] != 0:
    #     precision /= confusion_matrix[0][0] + confusion_matrix[1][0]
    # if confusion_matrix[0][0] + confusion_matrix[0][1] != 0:
    #     recall /= confusion_matrix[0][0] + confusion_matrix[0][1]

    # print '----------------------------'
    # print 'Perceptrons model evaluation'
    # print '----------------------------'
    # print 'Accuracy:  %.1f %%' % (accuracy * 100)
    # print 'Precision: %.1f %%' % (precision * 100)
    # print 'Recall:    %.1f %%' % (recall * 100)

    # a = 0
