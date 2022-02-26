# Solve the household problem in the last period

import numpy as np
from numba import njit, prange

# Local module
import utility

# Define the objective function
@njit
def obj_last_period(d,x,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x-d

    return -utility.func(c,d,par) # Minus as the code minimizes the function

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep[t]
    inv_marg_u_keep = sol.inv_marg_u_keep[t]
    c_keep = sol.c_keep[t]
    inv_v_adj = sol.inv_v_adj[t]
    inv_marg_u_adj = sol.inv_marg_u_adj[t]
    d_adj = sol.d_adj[t]
    c_adj = sol.c_adj[t]

    # a. keep
    for i_p in prange(par.Np): # Permanent income state
        for i_n in range(par.Nn): # Values of the durable
            for i_m in range(par.Nm): # Exogeneous cash on hand grid

                # i. states
                n = par.grid_n[i_n]
                m = par.grid_m[i_m]

                # Check why this step is required
                if m == 0: # forced c = 0 
                    c_keep[i_p,i_n,i_m] = 0
                    inv_v_keep[i_p,i_n,i_m] = 0
                    inv_marg_u_keep[i_p,i_n,i_m] = 0
                    continue
                
                # ii. optimal choice
                c_keep[i_p,i_n,i_m] = m

                # iii. optimal value
                v_keep = utility.func(c_keep[i_p,i_n,i_m],n,par) # Utility of choice
                inv_v_keep[i_p,i_n,i_m] = -1.0/v_keep # Compute minus inverse for numerical stability 
                inv_marg_u_keep[i_p,i_n,i_m] = 1.0/utility.marg_func(c_keep[i_p,i_n,i_m],n,par) # Inverse of the marginal utility

    # Solve adjuster problem
    # b. adj
    for i_p in prange(par.Np): # Permanent income states
        for i_x in range(par.Nx): # Cash on hand states
            
            # i. states
            x = par.grid_x[i_x]

            # Cannot do anything here
            if x == 0: # forced c = d = 0
                d_adj[i_p,i_x] = 0
                c_adj[i_p,i_x] = 0
                inv_v_adj[i_p,i_x] = 0
                inv_marg_u_adj[i_p,i_x] = 0
                continue

            # ii. optimal choices
            # d_low = 0 # Minimum value of housing
            # d_high = np.min([x, par.n_max])  #np.fmin(x,par.n_max) # Maximum available to purchase
            
            d_allow = par.d_grid[par.d_grid <= x] # values of housing that one can afford given x
            objective = obj_last_period(par.allow, x, par) # utility values on grid of allowed housing
            d_adj[i_p,i_x] = np.argmax(objective) # Find optimal value of housing
            c_adj[i_p,i_x] = x - d_adj[i_p,i_x] # Find implied consumption choice (using all funds left)
            # Just add house prices, then it should be fine

            # iii. optimal value
            v_adj = -obj_last_period(d_adj[i_p,i_x],x,par)
            inv_v_adj[i_p,i_x] = -1.0/v_adj
            inv_marg_u_adj[i_p,i_x] = 1.0/utility.marg_func(c_adj[i_p,i_x],d_adj[i_p,i_x],par)