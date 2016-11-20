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
        if min_val is None:
            self.min_val = -1
        else:
            self.min_val = min_val

        if max_val is None:
            self.max_val = 1
        else:
            self.max_val = max_val

    def compute(self, rand_obj):
        r = rand_obj.random()
        # r = 0.5 # for debug
        return r * (self.max_val - self.min_val) + self.min_val


class Binomial(RandomGenerator):

    def __init__(self, num_binary, prov_noise):
        super(Binomial, self).__init__('Binomial')
        self.set_param(num_binary, prov_noise)

    def set_param(self, num_binary, prov_noise):
        if num_binary is None:
            self.num_binary = 1
        else:
            self.num_binary = num_binary
        if prov_noise is None:
            self.prov_noise = 0
        else:
            self.prov_noise = prov_noise

    def compute(self, rand_obj):
        if self.prov_noise < 0 or self.prov_noise > 1:
            return 0

        cnt_val = 0
        for i in range(self.num_binary):
            rand_val = rand_obj.random()
            # rand_val = 0.5 # for debug
            if rand_val < self.prov_noise:
               cnt_val+=1

        return cnt_val
