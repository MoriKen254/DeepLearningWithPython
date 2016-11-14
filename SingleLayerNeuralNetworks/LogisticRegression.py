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

        y_err_arr = [[0] * self.dim_output_signal for i in range(min_batch_size)]

        # train with SGD
        # 1. calculate gradient of gradients_w, gradients_b
        ## loop for minibash size
        for n, (input_signal, teacher_label, y_err) in enumerate(zip(input_signals, teacher_labels, y_err_arr)):
            predicted_y_arr = self.output(input_signal)

            ## loop for output size
            for j, (predicted_y, teacher_elem, gradient_w) in enumerate(zip(predicted_y_arr, teacher_label, gradients_w)):
                # t_n - y_n : error between output and teacher
                y_err[j] = predicted_y - teacher_elem

                ## loop for input size
                for i, input_elem in enumerate(input_signal):
                    # dE/dw = - Sum{ (t_n - y_n) * x_n } ... (2.5.14)
                    gradient_w[i] += y_err[j] * input_elem

                # dE/db = - Sum{ t_n - y_n } ... (2.5.15)
                gradients_b[j] += y_err[j]

        # 2. update param
        for j, (gradient_w, gradient_b) in  enumerate(zip(gradients_w, gradients_b)):
            for i, gradient_w_elem in  enumerate(gradient_w):
                # w_(k+1) = w_(k) - eta * dE/dw = w_(k) - eta * Sum{ (t_n - y_n) * x_n } / min_batch_size ... (2.5.26)
                self.weights[j][i] -= learning_rate * gradient_w_elem / min_batch_size

            # b_(k+1) = b_(k) - eta * dE/db = b_(k) - eta * Sum{ (t_n - y_n) } / min_batch_size ... (2.5.27)
            self.biases[j] -= learning_rate * gradient_b / min_batch_size

        return y_err_arr


    # compute output layer with softmax
    def output(self, input_signals):
        pre_activations = [0] * self.dim_output_signal

        for j, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            for w, input_signal in zip(weight, input_signals):
                pre_activations[j] += w * input_signal

            pre_activations[j] += bias

        return Softmax.compute(pre_activations, self.dim_output_signal)


    def predict(self, input_signals):

        output_vals = self.output(input_signals) # activate input data through learned networks
        labels = [0] * len(output_vals) # output_vals is the probability, so cast it to label

        argmax = -1
        max_val = 0.

        for i, output_elem in enumerate(output_vals):
            if max_val < output_elem:
                max_val = output_elem
                argmax = i

        for i in range(len(labels)):
            if i == argmax:
                labels[i] = 1 # set 1 only to the maximum label
            else:
                labels[i] = 0 # otherwise 0

        return labels


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
    learning_rate = 0.2 # learning rate

    MIN_BATCH_SIZE = 50 # number of data in each minbatch
    CNT_MIN_BATCH = CNT_TRAIN_DATA / MIN_BATCH_SIZE

    train_input_data_set_min_batch = [[[0] * DIM_INPUT_SIGNAL for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    train_teacher_data_set_min_batch = [[[0] * DIM_OUTPUT_SIGNAL for j in range(MIN_BATCH_SIZE)]
                                      for k in range(CNT_MIN_BATCH)]
    min_batch_indexes = range(CNT_TRAIN_DATA)
    random.shuffle(min_batch_indexes)   # shuffle data index for SGD

    #
    # Create training data and test data for demo.
    #
    #   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
    #   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
    #   class 3 : x3 ~ N(  0.0, 1.0 ), y3 ~ N(  0.0, 1.0 )
    #

    rand_obj = random.Random()
    rand_obj.seed(1234)

    gaussian1 = GaussianDistribution(-2.0, 1.0, rand_obj)
    gaussian2 = GaussianDistribution(2.0, 1.0, rand_obj)
    gaussian3 = GaussianDistribution(0.0, 1.0, rand_obj)

    # data set in class 1
    cls_idx_prev = 0
    cls_idx_curr = 1
    CNT_TRAIN_DATA_PER_CLS = CNT_TRAIN_DATA/CNT_PATTERN
    CNT_TEST_DATA_PER_CLS = CNT_TEST_DATA/CNT_PATTERN
    for i in range(CNT_TRAIN_DATA_PER_CLS*cls_idx_prev, CNT_TRAIN_DATA_PER_CLS*cls_idx_curr):
        train_input_data_set[i][0] = gaussian1.get_random()
        train_input_data_set[i][1] = gaussian2.get_random()
        train_teacher_labels[i] = [1, 0, 0]
    for i in range(CNT_TEST_DATA_PER_CLS*cls_idx_prev, CNT_TEST_DATA_PER_CLS*cls_idx_curr):
        test_input_data_set[i][0] = gaussian1.get_random()
        test_input_data_set[i][1] = gaussian2.get_random()
        test_teacher_labels[i] = [1, 0, 0]

    # data set in class 2
    cls_idx_prev += 1
    cls_idx_curr += 1
    for i in range(CNT_TRAIN_DATA_PER_CLS*cls_idx_prev, CNT_TRAIN_DATA_PER_CLS*cls_idx_curr):
        train_input_data_set[i][0] = gaussian2.get_random()
        train_input_data_set[i][1] = gaussian1.get_random()
        train_teacher_labels[i] = [0, 1, 0]
    for i in range(CNT_TEST_DATA_PER_CLS*cls_idx_prev, CNT_TEST_DATA_PER_CLS*cls_idx_curr):
        test_input_data_set[i][0] = gaussian2.get_random()
        test_input_data_set[i][1] = gaussian1.get_random()
        test_teacher_labels[i] = [0, 1, 0]

    # data set in class 3
    cls_idx_prev += 1
    cls_idx_curr += 1
    for i in range(CNT_TRAIN_DATA_PER_CLS*cls_idx_prev, CNT_TRAIN_DATA_PER_CLS*cls_idx_curr):
        train_input_data_set[i][0] = gaussian3.get_random()
        train_input_data_set[i][1] = gaussian3.get_random()
        train_teacher_labels[i] = [0, 0, 1]
    for i in range(CNT_TEST_DATA_PER_CLS*cls_idx_prev, CNT_TEST_DATA_PER_CLS*cls_idx_curr):
        test_input_data_set[i][0] = gaussian3.get_random()
        test_input_data_set[i][1] = gaussian3.get_random()
        test_teacher_labels[i] = [0, 0, 1]

    # create minbatches with training data
    for i in range(CNT_MIN_BATCH):
        for j in range(MIN_BATCH_SIZE):
            idx = min_batch_indexes[i * MIN_BATCH_SIZE + j]
            train_input_data_set_min_batch[i][j] = train_input_data_set[idx]
            train_teacher_data_set_min_batch[i][j] = train_teacher_labels[idx]

    #
    # Build Logistic Regression model
    #

    # construct logistic regression
    classifier = LogisticRegression(DIM_INPUT_SIGNAL, DIM_OUTPUT_SIGNAL)

    for epoch in range(EPOCHS):   # training epochs
        for (train_input_data_min_batch, train_teacher_data_min_batch) in \
                zip(train_input_data_set_min_batch, train_teacher_data_set_min_batch):
            classifier.train(train_input_data_set, train_teacher_data_min_batch, MIN_BATCH_SIZE, learning_rate)

        print 'epoch = %.lf' % epoch

        learning_rate *= 0.95

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
        col = 0
        row = 0

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

    # if confusion_matrix[0][0] + confusion_matrix[1][0] != 0:
    #     precision /= confusion_matrix[0][0] + confusion_matrix[1][0]
    # if confusion_matrix[0][0] + confusion_matrix[0][1] != 0:
    #     recall /= confusion_matrix[0][0] + confusion_matrix[0][1]

    print '------------------------------------'
    print 'Logistic Regression model evaluation'
    print '------------------------------------'
    print 'Accuracy:  %.1f %%' % (accuracy * 100)
    print 'Precision:'
    for i, precision_elem in enumerate(precision):
        print 'class %d: %.1f %%' % (i+1, precision_elem * 100)
    print 'Recall:'
    for i, recall_elem in enumerate(recall):
        print 'class %d: %.1f %%' % (i+1, recall_elem * 100)

    a = 0
