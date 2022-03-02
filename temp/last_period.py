# Solve the household problem in the last period

import numpy as np
from numba import njit, prange

# Local module
import utility

# Need a name for housing in the last period (right now use d)

# Define the objective function
@njit
def obj_last_period(h,x,par):
    """ objective function in last period """
    
    # implied consumption after choosing housing
    c = x - par.p*h

    return -utility.func(c,h,par) # Minus as the code minimizes the function

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep[t]
    inv_marg_u_keep = sol.inv_marg_u_keep[t]
    c_keep = sol.c_keep[t]
    inv_v_adj = sol.inv_v_adj[t]
    inv_marg_u_adj = sol.inv_marg_u_adj[t]
    h_adj = sol.h_adj[t]
    c_adj = sol.c_adj[t]

    # a. keep (RENAME THIS)
    for i_y in prange(par.Ny): # Income states
        for i_h in range(par.Nh): # Values of the durable
            for i_m in range(par.Nm): # Exogeneous cash on hand grid

                # i. states
                h = par.grid_n[i_h]
                m = par.grid_m[i_m]

                # Check why this step is required
                if m == 0: # forced c = 0 
                    c_keep[i_y,i_h,i_m] = 0
                    inv_v_keep[i_y,i_h,i_m] = 0
                    inv_marg_u_keep[i_y,i_h,i_m] = 0
                    continue
                
                # ii. optimal choice
                c_keep[i_y,i_h,i_m] = m

                # iii. optimal value
                v_keep = utility.func(c_keep[i_y,i_h,i_m],h,par) # Utility of choice
                inv_v_keep[i_y,i_h,i_m] = -1.0/v_keep # Compute minus inverse for numerical stability 
                inv_marg_u_keep[i_y,i_h,i_m] = 1.0/utility.marg_func(c_keep[i_y,i_h,i_m],h,par) # Inverse of the marginal utility

    # Solve adjuster problem
    # b. adj
    for i_y in prange(par.Ny): # Income states
        for i_x in range(par.Nx): # Cash on hand states
            
            # i. states
            x = par.grid_x[i_x]

            # Cannot do anything here
            if x == 0: # forced c = d = 0
                h_adj[i_y,i_x] = 0
                c_adj[i_y,i_x] = 0
                inv_v_adj[i_y,i_x] = 0
                inv_marg_u_adj[i_y,i_x] = 0
                continue
            
            h_allow = par.grid_h[par.grid_h <= x / par.p] # values of housing that one can afford given x
            objective = obj_last_period(h, x, par) # utility values on grid of allowed housing
            h_adj[i_y,i_x] = np.argmax(objective) # Find optimal value of housing
            c_adj[i_y,i_x] = x - h_adj[i_y,i_x] # Find implied consumption choice (using all funds left)
            # Just add house prices, then it should be fine

            # iii. optimal value
            v_adj = -obj_last_period(h_adj[i_y,i_x],x,par)
            inv_v_adj[i_y,i_x] = -1.0/v_adj
            inv_marg_u_adj[i_y,i_x] = 1.0/utility.marg_func(c_adj[i_y,i_x],h_adj[i_y,i_x],par)