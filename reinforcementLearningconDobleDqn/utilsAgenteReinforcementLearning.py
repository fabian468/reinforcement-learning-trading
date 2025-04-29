# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:13:20 2025

@author: fabia
"""

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def price_format(n):
    n = float(n)
    if n < 0:
        return "- {0:.6f}".format(abs(n))
    else:
        return "{0:.6f}".format(abs(n))