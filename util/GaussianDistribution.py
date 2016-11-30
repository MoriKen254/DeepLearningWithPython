#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Masaru Morita

This software is released under the MIT License.
See LICENSE file included in this repository.
"""

import random
import math


class GaussianDistribution:

    def __init__(self, mean, var, rand_obj):
        if (var < 0):
            raise ValueError("save must be True if recurse is True")

        self.mean = mean
        self.var = var

        if rand_obj is None:
            rand_obj = random()

        self.rand_obj = rand_obj

    def get_random(self):
        rand_val = 0.0
        while rand_val == 0.0:
            rand_val = self.rand_obj.random()

        # Box-Muller's method
        coef = math.sqrt(-2.0 *  math.log(rand_val))

        if self.rand_obj.random() < 0.5:
            return coef * math.sin(2.0 * math.pi * self.rand_obj.random()) * self.var + self.mean

        return coef * math.cos(2.0 * math.pi * self.rand_obj.random()) * self.var + self.mean


