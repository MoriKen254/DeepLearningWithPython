#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

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
        self.func_name = 'Step'

    @classmethod
    def compute(self, x):
        if x >= 0:
            return 1
        else:
            return -1

