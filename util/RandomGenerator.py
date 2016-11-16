#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import random
import abc

class RandomGenerator(object):

    def __init__(self, obj_name):
        self.obj_name = obj_name

    @abc.abstractmethod
    def compute(self, rand_obj):
        pass

class Uniform(RandomGenerator):

    def __init__(self, min_val, max_val):
        super(Uniform, self).__init__('Uniform')
        self.set_param(min_val, max_val)

    def set_param(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def compute(self, rand_obj):
        return rand_obj.random() * (self.max_val - self.min_val) + self.min_val
