# -*- coding: utf-8 -*-
"""
Direct Integration Schemes used in Structural Dynamics
Created on Tue Jul  6 10:34:03 2021

@author: Alan Jason Correa
"""

import numpy as np



# Newmark Time Integration Scheme Incremental Form - Linear Acceleration
"""
Input : M, C, K, u_0, udot_0, uddot_0, f(t), h, n
Output: u(t), udot(t), uddot(t)

1. Initialize u = u_0, udot = udot_0, uddot = uddot_0, t = 0, e = f(t)
2. Calculate K_star:
    K_star = 6 * M / h**2 + 3 * C / h + K
3. Calculate Constants b1, b2:
    b1 = 3 * M + C * h / 2
    b2 = 6 * M / h + 3 * C
3. Initialize Solution vectors u_sol, udot_sol, uddot_sol, t_sol
4. For i in range(n):
    a. t = t + h
    b. Calculate R_star:
        -> delta_f = f(t) - e
        -> R_star = delta_f + b1 * uddot + b2 * udot
        -> e = e + delta_f
    c. Calculate delta_u:
        delta_u = inv(K_star) * R_star
        delta_udot = 3 * delta_u / h - 3 * udot - 0.5 * h * uddot
        delta_uddot = 6 * delta_u / h**2 - 6 * udot / h - 3 * uddot
    d. Update u, udot, uddot:
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
    e. Append u, udot, uddot, t -> u_sol, udot_sol, uddot_sol, t_sol

 5. Return u_sol, udot_sol and uddot_sol        

"""
def newmark_linear_acc_incremental(M, C, K, u_0, udot_0, uddot_0, f, h, n, 
                                   returnOnlyDisplacement = False):
    # Initialization
    u = u_0
    udot = udot_0
    uddot = uddot_0
    t = 0.0
    e = f(t) + M * uddot + C * udot + K * u  
    
    # Calculating K_star
    K_star = (6 / np.power(h,2)) * M  + (3 / h) * C + K
    
    # Calculating Constants b1, b2
    b1 = (6 / h) * M + 3 * C
    b2 = 3 * M + C * h / 2
    
    # Initializing Solution Vectors
    u_sol = np.array(u)
    udot_sol = np.array(udot)
    uddot_sol = np.array(uddot)
    t_sol = np.array(t)
    
    # Time Stepping
    for i in range(n):
        # New time Step
        t = t + h
        # Calculate R_star
        delta_f = f(t) - e 
        R_star = delta_f + b1 * udot + b2 * uddot
        e = e + delta_f
        # Calculate delta_u
        delta_u = R_star / K_star
        delta_udot = (3 / h) * delta_u - 3 * udot - 0.5 * h * uddot
        delta_uddot = (6 / np.power(h,2)) * delta_u - (6 / h) * udot - 3 * uddot
        # Update u, udot, uddot
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
        # Append Solution to Solution Vector
        u_sol = np.append(u_sol, u)
        udot_sol = np.append(udot_sol, udot)
        uddot_sol = np.append(uddot_sol, uddot)
        t_sol = np.append(t_sol, t)
    
    if (returnOnlyDisplacement):
        return t_sol, u_sol
    else:
        return t_sol, u_sol, udot_sol, uddot_sol

# Wilson Theta Time Integration Scheme
"""
Input : M, C, K, u_0, udot_0, uddot_0, f(t), h, n, theta
Output: u(t), udot(t), uddot(t)

1. Initialize u = u_0, udot = udot_0, uddot = uddot_0, t = 0, e = f(t), 
2. Calculate K_star:
    K_star = 6 * M / (theta*h)**2 + 3 * C / (theta*h) + K
3. Calculate Constants b1, b2:
    b1 = 2 * M + C * (h*theta) / 2
    b2 = 6 * M / (h*theta) + 2 * C
    b3 = 6 * M / (h*theta)**2 + 3 * C / (h*theta)
3. Initialize Solution vectors u_sol, udot_sol, uddot_sol, t_sol 
4. For i in range(n):
    a. t = t + h
    b. Calculate R_star:
        -> delta_f = f(t) - e
        -> R_star = e + theta* (delta_f) + b1 * uddot + b2 * udot  + b3 * u
        -> e = e + delta_f
    c. Calculate uddot_theta:
        u_theta = inv(K_star) * R_star
        uddot_theta = 6 / (h*theta)**2 * (u_theta - u) 
                     -6 / (h*theta) * udot - 2 * uddot
    d. Update uddot, udot, u:
        uddot_next = (1 - 1/theta) * uddot + (1/theta) * uddot_theta 
        udot_next = udot + h / 2 * (uddot + uddot_next)
        u_next = u + h * udot + h**2 / 6 * (uddot_next + 2 * uddot)
        uddot = uddot_next
        udot = udot_next
        u = u_next
    e. Append u, udot, uddot, t -> u_sol, udot_sol, uddot_sol, t_sol

 5. Return u_sol, udot_sol and uddot_sol   
"""
def wilson_theta (M, C, K, u_0, udot_0, uddot_0, f, h, n, theta, 
                  returnOnlyDisplacement = False):
    # Initialization
    u = u_0
    udot = udot_0
    uddot = uddot_0
    t = 0.0
    e = f(t) - M * uddot - C * udot - K * u 
    
    # Calculating K_star
    K_star = 6 / np.power((theta*h),2) * M + 3 / (theta*h) * C  + K
    
    # Calculating Constants b1, b2, b3
    b1 = 2 * M + (h*theta) / 2 * C
    b2 = 6 / (h*theta) * M  + 2 * C
    b3 = 6 / np.power((h*theta),2) * M + 3 / (h*theta) * C 
    
    # Initializing Solution Vectors
    u_sol = np.array(u)
    udot_sol = np.array(udot)
    uddot_sol = np.array(uddot)
    t_sol = np.array(t)
    
    # Time Stepping
    for i in range(n):
        # New time Step
        t = t + h
        # Calculate R_star
        delta_f = f(t) - e
        R_star = e + theta * delta_f + b1 * uddot + b2 * udot  + b3 * u
        e = e + delta_f
        # Calculate uddot_theta
        u_theta =  R_star / K_star
        uddot_theta = 6 / np.power((h*theta),2) * (u_theta - u) \
                    - 6 / (h*theta) * udot - 2 * uddot
        # Update u, udot, uddot
        uddot_next = (1 - 1/theta) * uddot + (1/theta) * uddot_theta 
        udot_next = udot + h / 2 * (uddot + uddot_next)
        u_next = u + h * udot + np.power(h,2) / 6 * (uddot_next + 2 * uddot)
        uddot = uddot_next
        udot = udot_next
        u = u_next
        # Append Solution to Solution Vector
        u_sol = np.append(u_sol, u)
        udot_sol = np.append(udot_sol, udot)
        uddot_sol = np.append(uddot_sol, uddot)
        t_sol = np.append(t_sol, t)
        
    if (returnOnlyDisplacement):
        return t_sol, u_sol
    else:
        return t_sol, u_sol, udot_sol, uddot_sol
    

# Wilson Time Integration Scheme Incremental Form - Theory 2
"""
Input : M, C, K, u_0, udot_0, uddot_0, f(t), h, n, theta
Output: u(t), udot(t), uddot(t)

1. Initialize u = u_0, udot = udot_0, uddot = uddot_0, t = 0, e = f(t)
2. Calculate M_bar, C_bar, K_bar
3. Calculate K_star:
    K_star = 6 * M_bar * theta / h**2 + 3 * C_bar * theta**2 / h + K * theta**3
4. Calculate Constants b1, b2:
    b1 = 3 * M_bar * theta + C_bar * (2*theta**2 - 2 + h /2) + K_bar * h**2 / 2 * (theta**3 - 2 * theta + 1)
    b2 = 6 * M_bar * theta / h + 3 * C_bar * theta**2 + K_bar * h * (theta**3 - 1)
3. Initialize Solution vectors u_sol, udot_sol, uddot_sol, t_sol
4. For i in range(n):
    a. t = t + h
    b. Calculate R_star:
        -> delta_f = f(t) - e
        -> R_star = delta_f + b1 * uddot + b2 * udot
        -> e = e + delta_f
    c. Calculate delta_u:
        delta_u = inv(K_star) * R_star
        delta_udot = 3 * delta_u / h - 3 * udot - 0.5 * h * uddot
        delta_uddot = 6 * delta_u / h**2 - 6 * udot / h - 3 * uddot
    d. Update u, udot, uddot:
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
    e. Append u, udot, uddot, t -> u_sol, udot_sol, uddot_sol, t_sol

 5. Return u_sol, udot_sol and uddot_sol        

"""
def wilson_incremental1(M, C, K, u_0, udot_0, uddot_0, f, h, n, theta, 
                                   returnOnlyDisplacement = False):
        # Calculating Modified Mass, Damping and Stiffness
    M_bar = M / theta
    C_bar = C / theta
    K_bar = K / theta
    
    # Initialization
    u = u_0
    udot = udot_0
    uddot = uddot_0
    t = 0.0
    e = f(t)
    
    # Calculating K_star
    K_star = (6* theta / np.power(h,2)) * M_bar \
            + (3 / h) * np.power(theta, 2) * C_bar \
            + K_bar * np.power(theta, 3)
    
    # Calculating Constants b1, b2
    b1 = 3 * M_bar * theta + C_bar * (2 * np.power(theta, 2)  - 2 + h /2) \
        + K_bar * np.power(h, 2) / 2 * (np.power(theta, 3) - 2 * theta + 1)
    b2 = 6 * M_bar * theta / h + 3 * C_bar * np.power(theta,2) \
        + K_bar * h * (np.power(theta, 3) - 1)
    
    # Initializing Solution Vectors
    u_sol = np.array(u)
    udot_sol = np.array(udot)
    uddot_sol = np.array(uddot)
    t_sol = np.array(t)
    
    # Time Stepping
    for i in range(n):
        # New time Step
        t = t + h
        # Calculate R_star
        delta_f = f(t) - e 
        R_star = delta_f + b1 * uddot + b2 * udot
        e = e + delta_f
        # Calculate delta_u
        delta_u = R_star / K_star
        delta_udot = (3 / h) * delta_u - 3 * udot - 0.5 * h * uddot
        delta_uddot = (6 / np.power(h,2)) * delta_u - (6 / h) * udot - 3 * uddot
        # Update u, udot, uddot
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
        # Append Solution to Solution Vector
        u_sol = np.append(u_sol, u)
        udot_sol = np.append(udot_sol, udot)
        uddot_sol = np.append(uddot_sol, uddot)
        t_sol = np.append(t_sol, t)
    
    if (returnOnlyDisplacement):
        return t_sol, u_sol
    else:
        return t_sol, u_sol, udot_sol, uddot_sol
    
# Wilson Time Integration Scheme Incremental Form - Theory 3
"""
Input : M, C, K, u_0, udot_0, uddot_0, f(t), h, n, theta
Output: u(t), udot(t), uddot(t)

1. Initialize u = u_0, udot = udot_0, uddot = uddot_0, t = 0, e = f(t)
2. Calculate M_bar, C_bar, K_bar
3. Calculate K_star:
    K_star = 6 * M_bar * theta / h**2 + 3 * C_bar * theta**2 / h + K * theta**3
4. Calculate Constants b1, b2:
    b1 = 3 * M_bar * theta + C_bar * (1.5 * theta**2 - theta) * h \
        + K_bar * h**2 / 2 * (theta**3 - theta**2)
    b2 = 6 * M_bar * theta / h + 3 * C_bar * theta**2 + K_bar * h * (theta**3 - theta)
3. Initialize Solution vectors u_sol, udot_sol, uddot_sol, t_sol
4. For i in range(n):
    a. t = t + h
    b. Calculate R_star:
        -> delta_f = f(t) - e
        -> R_star = delta_f + b1 * uddot + b2 * udot
        -> e = e + delta_f
    c. Calculate delta_u:
        delta_u = inv(K_star) * R_star
        delta_udot = 3 * delta_u / h - 3 * udot - 0.5 * h * uddot
        delta_uddot = 6 * delta_u / h**2 - 6 * udot / h - 3 * uddot
    d. Update u, udot, uddot:
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
    e. Append u, udot, uddot, t -> u_sol, udot_sol, uddot_sol, t_sol

 5. Return u_sol, udot_sol and uddot_sol        

"""
def wilson_incremental2(M, C, K, u_0, udot_0, uddot_0, f, h, n, theta, 
                                   returnOnlyDisplacement = False):
    # Calculating Modified Mass, Damping and Stiffness
    M_bar = M / theta
    C_bar = C / theta
    K_bar = K / theta
    
    # Initialization
    u = u_0
    udot = udot_0
    uddot = uddot_0
    t = 0.0
    e = f(t) + theta * (M_bar * uddot + C_bar * udot + K_bar * u)
    
    # Calculating K_star
    K_star = (6* theta / np.power(h,2)) * M_bar \
            + (3 / h) * np.power(theta, 2) * C_bar \
            + K_bar * np.power(theta, 3)
    
    # Calculating Constants b1, b2
    b1 = 3 * M_bar * theta \
        + C_bar * (1.5 * np.power(theta,2) - theta) * h \
        + K_bar * np.power(h,2) / 2 * (np.power(theta,3) - np.power(theta,2))
    b2 = 6 * M_bar * theta / h + 3 * C_bar * np.power(theta,2) \
        + K_bar * h * (np.power(theta,3) - theta)
    
    # Initializing Solution Vectors
    u_sol = np.array(u)
    udot_sol = np.array(udot)
    uddot_sol = np.array(uddot)
    t_sol = np.array(t)
    
    # Time Stepping
    for i in range(n):
        # New time Step
        t = t + h
        # Calculate R_star
        delta_f = f(t) - e 
        R_star = delta_f + b1 * uddot + b2 * udot
        e = e + delta_f
        # Calculate delta_u
        delta_u = R_star / K_star
        delta_udot = (3 / h) * delta_u - 3 * udot - 0.5 * h * uddot
        delta_uddot = (6 / np.power(h,2)) * delta_u - (6 / h) * udot - 3 * uddot
        # Update u, udot, uddot
        u = u + delta_u
        udot = udot + delta_udot
        uddot = uddot + delta_uddot
        # Append Solution to Solution Vector
        u_sol = np.append(u_sol, u)
        udot_sol = np.append(udot_sol, udot)
        uddot_sol = np.append(uddot_sol, uddot)
        t_sol = np.append(t_sol, t)
    
    if (returnOnlyDisplacement):
        return t_sol, u_sol
    else:
        return t_sol, u_sol, udot_sol, uddot_sol
    
    
        
        
