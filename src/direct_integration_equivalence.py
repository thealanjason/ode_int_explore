# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:06:50 2021

@author: Alan Jason Correa
"""

from math import pi, sqrt, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from direct_integration_schemes import *
from sdof_problems import *
%matplotlib auto

p = Problem1()

# Defining Intial Conditions 
t_0 = 0.0
u_0 = 1.0
udot_0 = 0.0
uddot_0 = 0.0

# Total time for Solution Space
T = 50 * pi
# Time Increment Interval 
h = 0.05
# Number of Solution Steps
n = int(T/h)
theta = 1.37

#%%
"""
Modifier Function to calculate Equivalent Mass, Damping and Stiffness Matrix
for use with the Newmark Incremental Linear Acceleration Scheme
Input: M, C, K, theta, h
Output : M_hat, C_hat, K_hat
"""

def modifier(M, C, K, theta, h):
    M_bar = M * theta + C * h / 2 * (np.power(theta,2) - 1)  \
            + K * np.power(h, 2) / 6 * (np.power(theta,3) - 3*theta + 2)
    C_bar = C + K * h * (theta - 1)
    K_bar = K
    print(M_bar, C_bar, K_bar)
    M_hat = M_bar / theta
    C_hat = C_bar / theta
    K_hat = K_bar / theta
    
    return M_hat, C_hat, K_hat
#%% Plotting

# Create New Plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 8, forward=True)
# Setting Up the Plot Properties
ax.set_xlabel(r'Time - $t$', size=12)
ax.set_ylabel(r'Displacement $u(t)$', size=12)
ax.tick_params(labelsize=12)
ax.title.set_text("Direct Time Integration Schemes")
plt.tight_layout()
plt.grid()


# Plotting the Solutions for SDOF Problem 
# 1. Newmark Linear Acceleration Incremental Scheme
# 2. Wilson Theta Scheme
line1, = ax.plot(*newmark_linear_acc_incremental(p.M, 
                                                  p.C, 
                                                  p.K, 
                                                  u_0, 
                                                  udot_0, 
                                                  uddot_0, 
                                                  p.f, 
                                                  h, n, 
                        returnOnlyDisplacement = True),
                  label='newmark_incremental',
                  linestyle='-', color='gray')

line2, = ax.plot(*wilson_theta(p.M, 
                               p.C, 
                               p.K, 
                               u_0, 
                               udot_0, 
                               uddot_0, 
                               p.f, 
                               h, n, 
                               theta,
                               returnOnlyDisplacement = True),
                 label='wilson_theta',
                 linestyle='dotted', color='r')

line3, = ax.plot(*wilson_incremental2(p.M, 
                                      p.C, 
                                      p.K, 
                                      u_0, 
                                      udot_0, 
                                      uddot_0, 
                                      p.f, 
                                      h, n, theta,
                        returnOnlyDisplacement = True),
                  label='newmark_incremental', 
                  linestyle='dashed', color='g')

# line3, = ax.plot(*wilson_incremental1(p.M, 
#                                       p.C, 
#                                       p.K, 
#                                       u_0, 
#                                       udot_0, 
#                                       uddot_0, 
#                                       p.f, 
#                                       h, n, theta,
#                         returnOnlyDisplacement = True),
#                   label='newmark_incremental', 
#                   linestyle='dotted', color='b')

# line4, = ax.plot(*newmark_linear_acc_incremental(*modifier(p.M, 
#                                                             p.C, 
#                                                             p.K, theta, h),
#                                       u_0, 
#                                       udot_0, 
#                                       uddot_0, 
#                                       p.f, 
#                                       h, n,
#                         returnOnlyDisplacement = True),
#                   label='newmark_incremental', 
#                   linestyle='dotted', color='b')