#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import abc
import math

class ActivationFunction(object):

    def __init__(self, activation_name):
        self.activation_name = activation_name

    @abc.abstractmethod
    def compute(self, x):
        pass

    @abc.abstractmethod
    def differentiate(self, x):
        pass


class Step(ActivationFunction):

    def __init__(self):
        super(Step, self).__init__('Step')

    def compute(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def differentiate(self, x):
        return 0


class Sigmoid(ActivationFunction):

    def __init__(self):
        super(Sigmoid, self).__init__('Sigmoid')

    def compute(self, x):
        den = 1. + pow(math.e, -x)
        return 1./den

    def differentiate(self, x):
        return x * (1. - x)

class Softmax:

    def __init__(self):
        self.func_name = 'Softmax'

    @staticmethod
    def compute(x_arr, dim_y):
        y_arr = [0] * dim_y
        max_val = 0.
        sum_val = 0.

        max_val = max(x_arr) # to prevent overflow

        for i, x in enumerate(x_arr):
            y_arr[i] = math.exp(x - max_val)
            sum_val += y_arr[i]

        for i, y in enumerate(y_arr):
            y_arr[i] /= sum_val

        return y_arr
