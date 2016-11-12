#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')
sys.path.append('../util')
from ActivationFunction import Step
import random

class Perceptrons:
    u"""
    Class for Perceptrons
    """

    def __init__(self, dim_input_signal):

        self.dim_input_signal = dim_input_signal
        self.weights = list([dim_input_signal])


    def train(self, input_signals, teacher_label, learning_rate):

        classified = 0
        sum_weighted_input = 0.0

        # check array size
        if len(input_signals) is not len(self.weights):
            return -1

        # check if the date is classified correctly
        for (weight, input_signal) in zip(self.weights, input_signals):
            sum_weighted_input += weight * input_signal * teacher_label

        # apply gradient descent method if the data is wrongly classified
        if sum_weighted_input > 0:
            classified = 1
        else:
            for (weight, input_signal) in zip(self.weights, input_signals):
                weight += learning_rate * input_signal * teacher_label

        return classified


    def predict(self, input_signals):

        pre_activation = 0.0

        # check array size
        if len(input_signals) is not len(self.weights):
            return

        for (weight, input_signal) in zip(self.weights, input_signals):
            pre_activation += weight * input_signal

        activation_function = Step()
        return activation_function.compute(a)


if __name__ == '__main__':
    CNT_TRAIN_DATA = 1000 # number of training data
    CNT_TEST_DATA = 200 # number of test data
    DIM_INPUT_SIGNAL = 2 # dimensions of input data

    # input data for training
    train_input_data_set = [[0 for i in range(DIM_INPUT_SIGNAL)] for j in range(CNT_TRAIN_DATA)]
    # output data (label) for training
    train_teacher_labels = [0 for i in range(CNT_TRAIN_DATA)]

    # input data for test
    test_input_data_set = [[0 for i in range(DIM_INPUT_SIGNAL)] for j in range(CNT_TRAIN_DATA)]
    # label for inputs
    test_teacher_labels = [0 for i in range(CNT_TRAIN_DATA)]
    # output data predicted by the model
    test_predict_teacher_labels = [0 for i in range(CNT_TRAIN_DATA)]

    EPOCHS = 2000   # maximum training epochs
    LEARNING_RATE = 1.0 # learning rate can be 1 in perceptrons

    #
    # Create training data and test data for demo.
    #
    # Let training data set for each class follow Normal (Gaussian) distribution here:
    #   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
    #   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
    #



