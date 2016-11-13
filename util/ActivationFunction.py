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
    def compute(self, x_arr, dim_x):
        y_arr = [0] * dim_x
        max = 0.
        sum = 0.

        max = max(x_arr) # to prevent overflow

        for i, x in enumerate(x_arr):
            y_arr[i] = math.exp(x - max)

        for i, y in enumerate(y_arr):
            y /= sum

        return y_arr
