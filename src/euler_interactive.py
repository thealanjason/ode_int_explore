# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:05:26 2021

@author: Alan Jason Correa
"""

#%% Imports

from math import pi, sqrt, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from euler_integration_schemes import *
from initial_value_problems import *
# %matplotlib auto

#%% Flags
# Flags for this script
interactive_plot = True


#%% IVP Defintion and Exact Solutuion
'''
Defining the Initial Value Problem
y' = f(t, y)
with initial conditions y(t0) = y0
'''

p = Problem2()

# Defining Intial Conditions 
t_init = 0.0
y_init = 1/sqrt(2)

#%% Parameter Definition
'''
Defining the Hyper Parameters for the given Problem
'''
# Scalar Multiplier for Stability
lamda = p.lamda
# Total time for Solution Space
T = 1 * pi 
# Time Increment Interval 
h = pi/50
# Number of Solution Steps
n = int(T/h)

#%% Plotting
# Create New Plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 8, forward=True)
# Setting Up the Plot Properties
ax.set_xlabel(r'Time - $t$', size=12)
ax.set_ylabel(r'Function - $y(t)$', size=12)
ax.tick_params(labelsize=12)
ax.title.set_text("Euler Time Integration Schemes")
plt.tight_layout()

if (interactive_plot):
    axcolor = 'lightgoldenrodyellow'
    plt.subplots_adjust(bottom=0.3)
    
    # Make a horizontal slider to control the h.
    ax_lamda = plt.axes([0.25, 0.15, 0.6, 0.025], facecolor=axcolor)
    lamda_slider = Slider(
        ax=ax_lamda,
        label='Lambda',
        valmin=-15.0,
        valmax= -0.5,
        valstep= -0.1,
        valinit=lamda,
    )
    
    ax_txbx_lamda = plt.axes([0.05, 0.15, 0.05, 0.04])
    lamda_txbx = TextBox(ax_txbx_lamda, label='', 
                         initial=round(float(lamda_slider.val), 3))
    
    # Make a horizontal slider to control the h.
    ax_timestep = plt.axes([0.25, 0.1, 0.6, 0.025], facecolor=axcolor)
    timestep_slider = Slider(
        ax=ax_timestep,
        label='Timestep [s]',
        valmin=pi/150,
        valmax=pi/4,
        valstep=pi/200,
        valinit=h,
    )
    
    ax_txbx_timestep = plt.axes([0.05, 0.1, 0.05, 0.04])
    timestep_txbx = TextBox(ax_txbx_timestep, label='',
                            initial=round(float(timestep_slider.val), 3))
    
    # Make a horizontal slider to control the h.
    ax_total_time = plt.axes([0.25, 0.05, 0.6, 0.025], facecolor=axcolor)
    total_time_slider = Slider(
        ax=ax_total_time,
        label='Total Time [s]',
        valmin=pi/8,
        valmax=5*pi,
        valstep=pi/15,
        valinit=T,
    )
    
    ax_txbx_total_time = plt.axes([0.05, 0.05, 0.05, 0.04])
    total_time_txbx = TextBox(ax_txbx_total_time, label='', 
                              initial= round(float(total_time_slider.val), 3))
    
    
    reset_ax = plt.axes([0.05, 0.2, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset',
                          color=axcolor, hovercolor='0.975')
    save_fig_ax = plt.axes([0.16, 0.2, 0.1, 0.04])
    save_fig_button = Button(save_fig_ax, 'Save Fig',
                             color=axcolor, hovercolor='0.975')

    def reset(event):
        reset_parameters()
        update_plots()
        ax.relim()
        ax.autoscale_view(scaley=True, scalex=True)
        fig.canvas.draw_idle()
    
    def update(val):
        global lamda, T, h
        update_textboxes()
        update_parameters()
        update_plots()
        ax.relim()
        ax.set_ylim(p.y_lims)
        ax.autoscale_view(scaley=True, scalex=True)
        fig.canvas.draw_idle()
    
    def update_parameters():
        global lamda, T, h, n
        p.lamda = lamda_slider.val
        T = total_time_slider.val
        h = timestep_slider.val
        n = int(T/h)
         
    def reset_parameters():
        global lamda, T, h, n
        lamda_slider.reset()
        total_time_slider.reset()
        timestep_slider.reset()
        p.lamda = lamda_slider.val
        T = total_time_slider.val
        h = timestep_slider.val
        n = int(T/h)  
    
    def update_plots():
        global lamda, T, h, n
        line1.set_data(*p.y_exact(y_init, t_init, t_init + T))
        line2.set_data(*implicit_euler_analytical(h, n, t_init, y_init,
                                                  p.f_implicit_analytical))
        line3.set_data(*implicit_euler_matrix(h, n, t_init, y_init,
                                              p.f_A, p.f_b))
        line4.set_data(*implicit_euler_iterative(h, n, t_init, y_init, p.f))
        line5.set_data(*explicit_euler(h, n, t_init, y_init, p.f))
        
    def save_fig(event):
        # Create New Plot
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(8, 5.5, forward=True)
        # Setting Up the Plot Properties
        ax1.set_xlabel(r'Time - $t$', size=12)
        ax1.set_ylabel(r'Function - $y(t)$', size=12)
        ax1.tick_params(labelsize=12)
        # ax1.title.set_text("Euler Time Integration Schemes")
        ax1.set_ylim(ax.get_ylim())
        plt.tight_layout()
        
        # Plotting the Solutions for ODE - Exact and Numerical Solutions
        line1, = ax1.plot(*p.y_exact(y0 = y_init, t0 = t_init, tn=t_init+T),
                         label='exact', color='black',linewidth='2')
        line2, = ax1.plot(*implicit_euler_analytical(h, n, t_init, y_init,
                                                     p.f_implicit_analytical),
                         label='implicit_analytic', markerfacecolor='none',
                         markeredgecolor='r', markersize='12', 
                         marker = 's', markeredgewidth=1.3,
                         linestyle='dotted', color='gray')
        line3, = ax1.plot(*implicit_euler_matrix(h, n, t_init, y_init,
                                                 p.f_A, p.f_b), 
                         label='implicit_matrix', markerfacecolor='none',
                         markeredgecolor='b', markersize='12',
                         marker = 'o', markeredgewidth=1,
                         linestyle='dotted', color='gray')
        line4, = ax1.plot(*implicit_euler_iterative(h, n, t_init, y_init, p.f),
                          label='implicit_iterative', markerfacecolor='none',
                         markeredgecolor='g', markersize='12', 
                         marker = '+', markeredgewidth=1.5,
                         linestyle='dotted', color='gray')
        line5, = ax1.plot(*explicit_euler(h, n, t_init, y_init, p.f), 
                         label='explicit', markerfacecolor='none',
                         markeredgecolor='deepskyblue', markersize='12', 
                         marker = 'o',  markeredgewidth=1,
                         linestyle='--', color='gray')
        
        
        ax1.legend(prop={'size': 12}, loc='lower left')
    
    def update_textboxes():
        lamda_txbx.set_val(round(lamda_slider.val, 3))
        timestep_txbx.set_val(round(timestep_slider.val, 3))
        total_time_txbx.set_val(round(total_time_slider.val, 3))
    
    def submit_lamda(text):
        lamda_slider.set_val(float(text))
        update(float(text))
        
    def submit_timestep(text):
        timestep_slider.set_val(float(text))
        update(float(text))

    def submit_total_time(text):
        total_time_slider.set_val(float(text))
        update(float(text))
        
    lamda_slider.on_changed(update)
    timestep_slider.on_changed(update)
    total_time_slider.on_changed(update)
    reset_button.on_clicked(reset)
    save_fig_button.on_clicked(save_fig)
    lamda_txbx.on_submit(submit_lamda)
    timestep_txbx.on_submit(submit_timestep)
    total_time_txbx.on_submit(submit_total_time)

# Plotting the Solutions for ODE - Exact and Numerical Solutions
line2, = ax.plot(*implicit_euler_analytical(h, n, t_init, y_init, 
                                            p.f_implicit_analytical),
                 label='imp_1', markerfacecolor='none',
                 markeredgecolor='r', markersize='12', 
                 marker = 's', markeredgewidth=1.3,
                 linestyle='dotted', color='gray')
line3, = ax.plot(*implicit_euler_matrix(h, n, t_init, y_init, p.f_A, p.f_b), 
                 label='imp_2', markerfacecolor='none',
                 markeredgecolor='b', markersize='12',
                 marker = 'o', markeredgewidth=1,
                 linestyle='dotted', color='gray')
line4, = ax.plot(*implicit_euler_iterative(h, n, t_init, y_init, p.f),
                  label='imp_3', markerfacecolor='none',
                 markeredgecolor='g', markersize='12', 
                 marker = '+', markeredgewidth=1.5,
                 linestyle='dotted', color='gray')
line5, = ax.plot(*explicit_euler(h, n, t_init, y_init, p.f), 
                 label='exp', markerfacecolor='none',
                 markeredgecolor='deepskyblue', markersize='12', 
                 marker = 'o',  markeredgewidth=1,
                 linestyle='--', color='gray')
line1, = ax.plot(*p.y_exact(y0 = y_init, t0 = t_init, tn=t_init+T),
                 label='exact', color='black',linewidth='2')

ax.relim()
ax.set_ylim(p.y_lims)
ax.autoscale_view(scaley=True, scalex=True)
leg = ax.legend(prop={'size': 12}, loc='lower left')
leg.set_draggable(True)


# %%
