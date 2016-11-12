#!/usr/bin/python
# -*- coding: utf-8 -*-


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

