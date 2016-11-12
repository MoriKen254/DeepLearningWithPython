#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')
sys.path.append('../util')
from ActivationFunction import Step


class Perceptrons:
    u"""
    Class for Perceptrons
    """

    def __init__(self, cnt_input_signal):

        self.cnt_input_signal = cnt_input_signal
        self.weights = list([cnt_input_signal])


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

        return step(pre_activation)

if __name__ == '__main__':

    activation_function = Step()
    a  = -1

    activation_function.compute(a)



