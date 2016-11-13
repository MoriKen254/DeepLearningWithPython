#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import math

class ActivationFunction:

    def __init__(self):
        self.func_name = 'ActivationFunction'

    @classmethod
    def compute(self, x):
        if x >= 0:
            return 1
        else:
            return -1


class Step(ActivationFunction):

    def __init__(self):
        super.__init__()
        self.func_name = 'Step'

    @classmethod
    def compute(self, x):
        if x >= 0:
            return 1
        else:
            return -1


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
