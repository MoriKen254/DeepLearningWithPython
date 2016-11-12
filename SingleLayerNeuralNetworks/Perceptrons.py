#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random

sys.path.append('../util')
from ActivationFunction import Step
from GaussianDistribution import GaussianDistribution

class Perceptrons:
    u"""
    Class for Perceptrons
    """

    def __init__(self, dim_input_signal):

        self.dim_input_signal = dim_input_signal
        self.weights = [0 for i in range(dim_input_signal)]


    def train(self, input_signals, teacher_label, learning_rate):

        classified = 0
        sum_weighted_input = 0.0

        # check array size
        if len(input_signals) is not len(self.weights):
            print('dimension of input signal is different from that of weight')
            return -1

        # check if the date is classified correctly
        for (weight, input_signal) in zip(self.weights, input_signals):
            sum_weighted_input += weight * input_signal * teacher_label

        # apply gradient descent method if the data is wrongly classified
        if sum_weighted_input > 0:
            classified = 1
        else:
            for i, input_signal in enumerate(input_signals):
                self.weights[i] += learning_rate * input_signal * teacher_label

        return classified


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
    CNT_TRAIN_DATA = 1000 # number of training data
    CNT_TEST_DATA = 200 # number of test data
    DIM_INPUT_SIGNAL = 2 # dimensions of input data

    # input data for training
    train_input_data_set = [[0 for i in range(0, DIM_INPUT_SIGNAL)] for j in range(0, CNT_TRAIN_DATA)]
    # output data (label) for training
    train_teacher_labels = [0 for i in range(0, CNT_TRAIN_DATA)]

    # input data for test
    test_input_data_set = [[0 for i in range(0, DIM_INPUT_SIGNAL)] for j in range(0, CNT_TEST_DATA)]
    # label for inputs
    test_teacher_labels = [0 for i in range(0, CNT_TEST_DATA)]
    # output data predicted by the model
    test_predict_output_labels = [0 for i in range(0, CNT_TEST_DATA)]

    EPOCHS = 10 #2000   # maximum training epochs
    LEARNING_RATE = 1.0 # learning rate can be 1 in perceptrons

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

    # data set in class 1
    # for i, train_input_data in enumerate(train_input_data_set):
    for i in range(0, CNT_TRAIN_DATA/2):
        train_input_data_set[i][0] = gaussian1.get_random()
        train_input_data_set[i][1] = gaussian2.get_random()
        train_teacher_labels[i] = 1
    for i in range(0, CNT_TEST_DATA/2):
        test_input_data_set[i][0] = gaussian1.get_random()
        test_input_data_set[i][1] = gaussian2.get_random()
        test_teacher_labels[i] = 1

    # data set in class 2
    for i in range(CNT_TRAIN_DATA/2, CNT_TRAIN_DATA):
        train_input_data_set[i][0] = gaussian1.get_random()
        train_input_data_set[i][1] = gaussian2.get_random()
        train_teacher_labels[i] = -1
    for i in range(CNT_TEST_DATA/2, CNT_TEST_DATA):
        test_input_data_set[i][0] = gaussian1.get_random()
        test_input_data_set[i][1] = gaussian2.get_random()
        test_teacher_labels[i] = -1

    #
    # build Perceptron model
    #
    epoch = 0   # training epochs

    # construct perceptrons
    classifier = Perceptrons(DIM_INPUT_SIGNAL)

    # train models
    while True:
        classified_sum = 0

        for (train_input_data, train_teacher_label) in zip(train_input_data_set, train_teacher_labels):
            classified_sum += classifier.train(train_input_data, train_teacher_label, LEARNING_RATE)

        if classified_sum == CNT_TRAIN_DATA:
            break

        epoch += 1
        if (epoch > EPOCHS):
            break

    # test
    for i, test_input_data in enumerate(test_input_data_set):
        test_predict_output_labels[i] = classifier.predict(test_input_data)

    #
    # Evaluate the model
    #
    confusion_matrix = [[0 for i in range(0, 2)] for j in range(0, 2)]
    accuracy = 0.
    precision = 0.
    recall = 0.

    for test_predict_output_label in test_predict_output_labels:
        if test_predict_output_label > 0:
            if(test_teacher_labels > 0):
                accuracy += 1
                precision += 1
                recall += 1
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][0] += 1
        else:
            if(test_teacher_labels > 0):
                confusion_matrix[0][1] += 1
            else:
                accuracy += 1
                confusion_matrix[1][1] += 1

    accuracy /= CNT_TEST_DATA
    precision /= confusion_matrix[0][0] + confusion_matrix[1][0]
    recall /= confusion_matrix[0][0] + confusion_matrix[0][1]

    print("----------------------------")
    print("Perceptrons model evaluation")
    print("----------------------------")
    print("Accuracy:  %.1f %%\n", accuracy * 100)
    print("Precision: %.1f %%\n", precision * 100)
    print("Recall:    %.1f %%\n", recall * 100)

    a = 0