# -*- coding: utf-8 -*-
"""
Python Script for Euler Time Integration Schemes
"""


# Importing Necessary Libraries
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import ipywidgets as widgets
# %matplotlib auto

# Defining the Hyper Parameters for the given Problem
pi = math.pi
lmda = -5.0

# Total time for Solution Space
run_time = 1 * pi 
# Time Increment Interval 
h = pi/25
# Number of Solution Steps
n = int(run_time/h)

def f(y, t):
    global lmda
    return (lmda * ( y- math.sin(t)) + math.cos(t))

# Defining Intial Conditions 
t_init = 0.0
y_init = 1/math.sqrt(2)

def y_exact(t):
    global y_init, t_init
    v1 = np.multiply((y_init - np.sin(t_init)), np.exp(lmda*(t-t_init)))
    v2 = np.sin(t)
    return v1 + v2

def exact_solution(h, n, y_exact):
    t_analytical = np.linspace(t_init, t_init + n * h, n * 10)
    y_analytical = y_exact(t_analytical)
    return (y_analytical, t_analytical)

(y_analytical, t_analytical) = exact_solution(h, n, y_exact)

fig1 = plt.subplot()
fig1.plot(t_analytical, y_analytical, 
          label='exact', color='gray', linewidth='3')

fig1.legend(prop={'size': 14})
fig1.set_xlabel('t', size=16)
fig1.set_ylabel('y', size=16)
fig1.tick_params(labelsize=14)

plt.show()

# Explicit Euler Method

def explicit_euler(h, n):
    #Setting Initial Conditions for the Initial Value Problem
    y_0 = y_init
    t_0 = t_init
    
    #Initializing Solution Arrays with Initial Values
    y_explicit = np.array(y_0)
    t_explicit = np.array(t_0)
    
    y_next = y_0
    t_next = t_0
    
    # Marching Forward in time
    for i in range(n):
        
        # Calculate Solution at Next Time Step
        y_next = y_next + h * (f(y_next, t_next))
        
        # Increment Time Step 
        t_next = t_next + h
        
        # Storing Values for Plotting
        y_explicit = np.append(y_explicit, y_next)
        t_explicit = np.append(t_explicit, t_next)
        
    return y_explicit, t_explicit


# Implicit Euler Method

# Setting Intial Conditions for the Initial Value Problem
y_0 = y_init
t_0 = t_init


def implicit_euler_analytical(h, n):
    """
    Method 1  : Using Analytically Calculated Derivative of y at t_n+1
    Stability : Unconditionally Stable
    Drawback  : Derivative at t_n+1 needs to be manually calculated for each function

    # See calculation of derivative at next time step here
    # http://people.bu.edu/andasari/courses/numericalpython/Week5Lecture8/ExtraHandout_Lecture8.pdf
    """ 
    
    # Intializing Solution Arrays
    global y_implicit_1
    global t_implicit_1
    y_implicit_1 = np.array(y_0)
    t_implicit_1 = np.array(t_0)
    
    # Initializing Parameters for Time Integration
    y_next = y_0
    t_next = t_0
    calculations = 0
    
    # Marching forward in time
    for i in range(n):
        calculations += 1
        
        # Increment Time
        t_next = t_next + h
        
        # Calculate Solution at Next Time Step
        y_next = y_next + h * (1 /(1 - h * lmda) * f(y_next, t_next))
        
        # Storing Values for Plotting
        y_implicit_1 = np.append(y_implicit_1, y_next)
        t_implicit_1 = np.append(t_implicit_1, t_next)
          
    return (y_implicit_1, t_implicit_1)

def implicit_euler_matrix(h, n):
    """
    Method 2  : Using Matrix Methods for the Entire Solution Space
    Stability : Unconditionally Stable
    Drawback  : Matrix Inversion needs to be performed
    """ 

    # In this method we solve for the system of Linear Equations
    # A y = b -----> y = inv(A) * b
    
    # Initializing Solution Arrays with Intitial Values
    global y_implicit_2
    global t_implicit_2
    y_implicit_2 = np.array(y_0)
    t_implicit_2 = np.array(t_0)

    # Initializing Parameters for Solution
    t_next = t_0
    y_next = y_0
    b = np.zeros(n)
    A = np.zeros(n*n).reshape(n, n)

    # Setting Up Vector b
    for i in range(n):
        t_next = t_next + h
        t_implicit_2 = np.append(t_implicit_2, t_next)
        b_n = -h * (lmda * math.sin(t_next) - math.cos(t_next))
        if i==0:
            b[i] = b_n + y_0
        else:
            b[i] = b_n
    
    # Setting Up Matrix A
    for i in range(n):
        A[i,i] = 1 - h * lmda
        if i > 0:
            A[i, i-1] = -1
    
    # Matrix Inversion
    A_inv = np.linalg.inv(A)
    
    # Calculating Solution Space 
    y_sol = A_inv @ b
    
    # Adding initial Value to Solution Space
    y_implicit_2 = np.append(y_implicit_2, y_sol)
    
    return (y_implicit_2, t_implicit_2)


def implicit_euler_iterative(h,n):
    """
    Method 3  : Using Iterative Predictor-Corrector Methods
    Stability : Conditionally Stable due to numerical approximation errors
    Drawback  : Convergence is not guaranteed
    """ 
    # Initializing Solution Arrays
    global y_implicit_3
    global t_implicit_3
    y_implicit_3 = np.array(y_0)
    t_implicit_3 = np.array(t_0)

    # Initializing Parameters for Time Integration
    y_next = y_0
    t_next = t_0
    convergence = True
    calculations = 0
    tolerance = 1e-10
    div_limit = 10000

    # Marching Forward in time
    for i in range(n):
        if (convergence):
            # Initialize Error for Convergence testing
            error = 1
            iteration = 0

            # Predictor - Explicit Euler
            y_next_iter = y_next + h * (f(y_next, t_next))

            # Increment Time
            t_next = t_next + h

            # Iterative Convergence using Corrector
            while(True and convergence):

                iteration +=1
                calculations +=1

                y_prev_iter = y_next_iter

                # Corrector 
                y_next_iter = y_next + h * f(y_next_iter, t_next)

                # Calculate Error
                error = abs(y_next_iter - y_prev_iter)

                # When solution converges 
                if(error < tolerance):
                    # Use converged Value of Solution for next Time Step
                    y_next = y_next_iter

                    # Store Values for Plotting
                    y_implicit_3 = np.append(y_implicit_3, y_next)
                    t_implicit_3 = np.append(t_implicit_3, t_next)
                    break

                # When solution diverges
                if(iteration > div_limit):
                    convergence = False
                    print("Implicit Euler - Iterative Solution Diverged!")
                    
    return (y_implicit_3, t_implicit_3)
    
# Calculate Solutions
(y_explicit, t_explicit) = explicit_euler(h, n)
(y_implicit_1, t_implicit_1) = implicit_euler_analytical(h, n)
(y_implicit_2, t_implicit_2) = implicit_euler_matrix(h, n)
(y_implicit_3, t_implicit_3) = implicit_euler_iterative(h, n)

# Create New Plot
fig = plt.figure()
ax = fig.add_subplot(111)

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

plt.subplots_adjust(bottom=0.3)

# Make a horizontal slider to control the h.
axfreq = plt.axes([0.2, 0.15, 0.7, 0.025], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Lambda',
    valmin=0.1,
    valmax=30,
    valinit=3,
)

# Make a horizontal slider to control the h.
axfreq = plt.axes([0.2, 0.1, 0.7, 0.025], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Timestep [s]',
    valmin=0.1,
    valmax=30,
    valinit=3,
)

# Make a horizontal slider to control the h.
axfreq = plt.axes([0.2, 0.05, 0.7, 0.025], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Total Time [s]',
    valmin=0.1,
    valmax=30,
    valinit=3,
)


# Plotting the Solutions for ODE - Exact and Numerical Solutions
line1, = ax.plot(t_analytical, y_analytical, label='exact', color='gray',linewidth='3')
line2, = ax.plot(t_implicit_1, y_implicit_1, label='impicit_analytical', color='gold', markersize='10', marker='s', linestyle='dotted', linewidth='2')
line3, = ax.plot(t_implicit_2, y_implicit_2, label='impicit_matrix', color='mediumpurple', markersize='10', marker='*', linestyle='-.', linewidth='2')
line4, = ax.plot(t_implicit_3, y_implicit_3, label='impicit_iterative', color='tomato', markersize='10', marker='x', linestyle='-.', linewidth='2')
line5, = ax.plot(t_explicit  , y_explicit  , label='explicit', color='deepskyblue', markersize='10', marker = '+', linestyle='--', linewidth='2')

# Setting Up the Plot Properties
ax.legend(prop={'size': 14}, loc='lower right')
ax.set_xlabel('t', size=16)
ax.set_ylabel('y', size=16)
ax.tick_params(labelsize=14)


# Update Function for Interactive Plot
def update(Time_Step=pi/20, lamda = -2, Total_Time = pi):
    global lmda, run_time, h
    # Update Values from Sliders for Solutions
    lmda = lamda
    h = Time_Step
    run_time = Total_Time
    n = int(run_time/h)
    print(" ", end='\r')
    
    # Calculate New Solutions
    (y_analytical, t_analytical) = exact_solution(h, n, y_exact)
    (y_explicit, t_explicit) = explicit_euler(h, n)
    (y_implicit_1, t_implicit_1) = implicit_euler_analytical(h, n)
    (y_implicit_2, t_implicit_2) = implicit_euler_matrix(h, n)
    (y_implicit_3, t_implicit_3) = implicit_euler_iterative(h, n)
    
    # Update Plotting Data with New Solutions
    line1.set_ydata(y_analytical)
    line1.set_xdata(t_analytical)
    
    line2.set_ydata(y_implicit_1)
    line2.set_xdata(t_implicit_1)
    
    line3.set_ydata(y_implicit_2)
    line3.set_xdata(t_implicit_2)
    
    line4.set_ydata(y_implicit_3)
    line4.set_xdata(t_implicit_3)
    
    line5.set_ydata(y_explicit)
    line5.set_xdata(t_explicit)
    
    # Scaling Settings
    ax.relim()
    plot_ylim = ax.get_ylim()
    if(plot_ylim[0] > -1.2): 
        ax.autoscale_view(scaley=True, scalex=True)
    else:
        ax.autoscale_view(scaley=False, scalex=True)
    
    # Update Plot
    plt.show()
    
# Show Interactive Plot    
widgets.interact(update, Time_Step = (pi/150, pi/4, pi/200), lamda=(-15.0, -0.1, 0.5), Total_Time = (pi/8, 5*pi, pi/15));
