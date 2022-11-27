# -*- coding: utf-8 -*-
"""
Problem Classes
Created on Fri Jun 25 10:01:27 2021

@author: Alan Jason Correa
"""

import numpy as np

class Problem1:
    def __init__(self):
        self.lamda = -4.0
        self.y_lims=[]
        print("Problem: y'(t) = lamda * (y(t) - sin(t)) + cos(t)")
    
    # Forcing Function
    def g(self, t):
        return - self.lamda * np.sin(t) + np.cos(t)
        
    # Defining y' 
    def f(self, y, t):
        return self.lamda * y + self.g(t)
    
    # Defining y'implicit analytical
    def f_implicit_analytical(self, h, y, t):
        return self.f(y, t) /(1.0 - h * self.lamda)
    
    # Exact Solution for comparison
    def y_exact(self, y0, t0, tn):
        t_sol = np.linspace(t0, tn, 1000)
        v1 = np.multiply((y0 - np.sin(t0)), np.exp(self.lamda*(t_sol-t0)))
        v2 = np.sin(t_sol)
        y_sol = v1 + v2
        self.y_lims = [np.min(y_sol)-0.05, np.max(y_sol)+0.05]
        return t_sol, y_sol
    
    def f_A(self, n, h):
        A = np.zeros(n*n).reshape(n, n)
        for i in range(n):
            A[i,i] = 1 - h * self.lamda
            if i > 0:
                A[i, i-1] = -1
        return A
                
    def f_b(self, n, h, t0, y0):
        t_next = t0
        b = np.zeros(n)
        for i in range(n):
            t_next = t_next + h
            b_n = h * self.g(t_next)
            if i==0:
                b[i] = b_n + y0
            else:
                b[i] = b_n
        return b
    
    
class Problem2:
    def __init__(self):
        self.lamda = -4.0
        self.y_lims = []
        print("")
    
    # Forcing Function
    def g(self, t):
        return np.sin(t)
        
    # Defining y' 
    def f(self, y, t):
        return self.lamda * y + self.g(t)
    
    # Defining y'implicit analytical
    def f_implicit_analytical(self, h, y, t):
        return self.f(y, t) /(1.0 - h * self.lamda)
    
    
    # Exact Solution for comparison
    def y_exact(self, y0, t0, tn):
        t_sol = np.linspace(t0, tn, 1000)
        v1 = (y0 + 1/(1+np.power(self.lamda, 2))) * np.exp(self.lamda * t_sol)
        v2 = -1/(1+np.power(self.lamda, 2)) * (self.lamda * np.sin(t_sol) + 
                                               np.cos(t_sol))
        y_sol = v1 + v2
        self.y_lims = [np.min(y_sol)-0.05, np.max(y_sol) +0.05]
        return t_sol, y_sol
    
    def f_A(self, n, h):
        A = np.zeros(n*n).reshape(n, n)
        for i in range(n):
            A[i,i] = 1 - h * self.lamda
            if i > 0:
                A[i, i-1] = -1
        return A
                
    def f_b(self, n, h, t0, y0):
        t_next = t0
        b = np.zeros(n)
        for i in range(n):
            t_next = t_next + h
            b_n = h * self.g(t_next)
            if i==0:
                b[i] = b_n + y0
            else:
                b[i] = b_n
        return b

class Problem3:
    def __init__(self):
        self.lamda = -4.0
        self.y_lims = []
        print("")
    
    # Forcing function
    def g(self, t):
        return np.sin(t) + np.cos(t)
        
    # Defining y' 
    def f(self, y, t):
        return self.lamda * y + self.g(t)
    
    # Defining y'implicit analytical
    def f_implicit_analytical(self, h, y, t):
        return self.f(y, t) /(1.0 - h * self.lamda)
    
    
    # Exact Solution for comparison
    def y_exact(self, y0, t0, tn):
        t_sol = np.linspace(t0, tn, 1000)
        v1 = (y0 + (self.lamda+1)/(1 +np.power(self.lamda, 2))) * \
             np.exp(self.lamda * t_sol)
        v2 = np.sqrt(2)/(1+np.power(self.lamda, 2)) * \
            (-self.lamda * np.sin(t_sol+np.pi/4) - np.cos(t_sol + np.pi/4))
        y_sol = v1 + v2
        self.y_lims = [np.min(y_sol)-0.05, np.max(y_sol) +0.05]
        return t_sol, y_sol
    
    def f_A(self, n, h):
        A = np.zeros(n*n).reshape(n, n)
        for i in range(n):
            A[i,i] = 1 - h * self.lamda
            if i > 0:
                A[i, i-1] = -1
        return A
                
    def f_b(self, n, h, t0, y0):
        t_next = t0
        b = np.zeros(n)
        for i in range(n):
            t_next = t_next + h
            b_n = h * self.g(t_next)
            if i==0:
                b[i] = b_n + y0
            else:
                b[i] = b_n
        return b
    
class Problem4:
    def __init__(self):
        self.lamda = -4.0
        self.y_lims = []
        print("")
    
    # Forcing function
    def g(self, t):
        return np.sin(t) + np.cos(2*t)
        
    # Defining y' 
    def f(self, y, t):
        return self.lamda * y + self.g(t)
    
    # Defining y'implicit analytical
    def f_implicit_analytical(self, h, y, t):
        return self.f(y, t) /(1.0 - h * self.lamda)
    
    
    # Exact Solution for comparison
    def y_exact(self, y0, t0, tn):
        t_sol = np.linspace(t0, tn, 1000)
        v1 = (y0 + (np.power(self.lamda, 3) + np.power(self.lamda, 2) + \
                    np.power(self.lamda, 1) + 4) / \
              (np.power(self.lamda, 4) + 5*np.power(self.lamda, 2) + 4)) * \
             np.exp(self.lamda * t_sol)
        v2 = (1 / (4+np.power(self.lamda, 4)+ 5*np.power(self.lamda, 2))) * \
            ((-np.power(self.lamda, 3) * (np.sin(t_sol) + np.cos(2*t_sol))) + \
             (np.power(self.lamda, 2) * (2*np.sin(2*t_sol) - np.cos(t_sol))) + \
             (-self.lamda * (4*np.sin(t_sol) + np.cos(2*t_sol))) + \
             (2 * (np.sin(2*t_sol) - 2 * np.cos(t_sol))))
        y_sol = v1 + v2
        self.y_lims = [np.min(y_sol)-0.05, np.max(y_sol) +0.05]
        return t_sol, y_sol
    
    def f_A(self, n, h):
        A = np.zeros(n*n).reshape(n, n)
        for i in range(n):
            A[i,i] = 1 - h * self.lamda
            if i > 0:
                A[i, i-1] = -1
        return A
                
    def f_b(self, n, h, t0, y0):
        t_next = t0
        b = np.zeros(n)
        for i in range(n):
            t_next = t_next + h
            b_n = h * self.g(t_next)
            if i==0:
                b[i] = b_n + y0
            else:
                b[i] = b_n
        return b
