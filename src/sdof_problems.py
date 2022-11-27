# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 19:50:54 2021

@author: Alan Jason Correa
"""

import numpy as np

class Problem1:
    def __init__(self):
        self.M = 82.0
        self.C = 10.0
        self.K = 160.0
        
    # Forcing Function
    def f(self, t):
        if t > 20 * np.pi:
            return 0.0
        else:
            return 40 * np.exp(-0.01 * t) * np.sin(1*t)