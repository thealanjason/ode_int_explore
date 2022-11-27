# -*- coding: utf-8 -*-
"""
Created on Mon April 12 10:54:37 2021

@author: Alan Jason Correa
"""

import numpy as np

def explicit_euler(h, n, t0, y0, f):
    
    #Initializing Solution Arrays with Initial Values
    y_sol = np.array(y0)
    t_sol = np.array(t0)
    
    y_next = y0
    t_next = t0
    
    # Marching Forward in time
    for i in range(n):
        
        # Calculate Solution at Next Time Step
        y_next = y_next + h * (f(y_next, t_next))
        
        # Increment Time Step 
        t_next = t_next + h
        
        # Storing Values for Plotting
        y_sol = np.append(y_sol, y_next)
        t_sol = np.append(t_sol, t_next)
        
    return t_sol, y_sol


def implicit_euler_analytical(h, n, t0, y0, f):
    """
    Method 1  : Using Analytically Calculated Derivative of y at t_n+1
    Stability : Unconditionally Stable
    Drawback  : Derivative at t_n+1 needs to be manually calculated 
            for each function

    # See calculation of derivative at next time step here
    # http://people.bu.edu/andasari/courses/numericalpython/Week5Lecture8/
    ExtraHandout_Lecture8.pdf
    """ 
    
    # Intializing Solution Arrays
    y_sol = np.array(y0)
    t_sol = np.array(t0)
    
    # Initializing Parameters for Time Integration
    y_next = y0
    t_next = t0
    
    # Marching forward in time
    for i in range(n):
        # Increment Time
        t_next = t_next + h
        
        # Calculate Solution at Next Time Step
        y_next = y_next + h * f(h, y_next, t_next)
        
        # Storing Values for Plotting
        y_sol = np.append(y_sol, y_next)
        t_sol = np.append(t_sol, t_next)
          
    return t_sol, y_sol

def implicit_euler_matrix(h, n, t0, y0, f_A, f_b):
    """
    Method 2  : Using Matrix Methods for the Entire Solution Space
    Stability : Unconditionally Stable
    Drawback  : Matrix Inversion needs to be performed
    
    # In this method we solve for the system of Linear Equations
    # A y = b -----> y = inv(A) * b
    """ 
    
    # Initializing Solution Arrays with Intitial Values
    y_sol = np.array(y0)
    t_sol = np.array(t0)

    # Setting Up t_sol
    t_next = t0
    for i in range(n):
        t_next = t_next + h
        t_sol = np.append(t_sol, t_next)

    # Setting Up Vector b
    b = f_b(n, h, t0, y0)
    
    # Setting Up Matrix A
    A = f_A(n, h)
    
    # Matrix Inversion
    A_inv = np.linalg.inv(A)
    
    # Calculating Solution Space 
    y_next_all = A_inv @ b
    
    # Adding initial Value to Solution Space
    y_sol = np.append(y_sol, y_next_all)
    
    return t_sol, y_sol

def implicit_euler_iterative(h, n, t0, y0, f, tol=1e-5, max_iter=10000):
    """
    Method 3  : Using Iterative Predictor-Corrector Methods
    Stability : Conditionally Stable due to numerical approximation errors
    Drawback  : Convergence is not guaranteed
    """ 
    # Initializing Solution Arrays
    y_sol = np.array(y0)
    t_sol = np.array(t0)

    # Initializing Parameters for Time Integration
    y_next = y0
    t_next = t0
    convergence = True

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

                y_prev_iter = y_next_iter

                # Corrector 
                y_next_iter = y_next + h * f(y_next_iter, t_next)

                # Calculate Error
                error = abs(y_next_iter - y_prev_iter)

                # When solution converges 
                if(error < tol):
                    # Use converged Value of Solution for next Time Step
                    y_next = y_next_iter

                    # Store Values for Plotting
                    y_sol = np.append(y_sol, y_next)
                    t_sol = np.append(t_sol, t_next)
                    break

                # When solution diverges
                if(iteration > max_iter):
                    convergence = False
                    print("Implicit Euler - Iterative Solution Diverged!")
                    
    return t_sol, y_sol