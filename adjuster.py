from re import S
import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp

# local modules
import utility

##########
# Adjust #
##########

@njit #(parallel=True)
def solve_adj(t,sol,par,ph):
    """solve bellman equation for adjusters using value function iteration"""

    # a. Unpack output
    inv_v = sol.inv_v_adj[t]
    inv_marg_u = sol.inv_marg_u_adj[t]
    d = sol.d_adj[t]
    c = sol.c_adj[t]

    # b. Unpack input
    inv_v_keep = sol.inv_v_keep[t]
    c_keep = sol.c_keep[t]
    grid_n = par.grid_n
    grid_m = par.grid_m
    d_ubar = par.d_ubar
    alpha = par.alpha
    rho = par.rho

    # c. Container for value of housing choice
    value_of_choice = np.zeros(shape=par.Nn)

    # d. Loop over outer states
    for i_pb in range(par.Npb):
        for i_y in range(par.Ny):
            for i_x in range(par.Nx):
                
                # i. Cash-on-hand
                x = par.grid_x[i_x]
                
                # ii. Force c = d = 0 if x = 0
                if i_x == 0:
                    d[i_pb,i_y,i_x] = 0
                    c[i_pb,i_y,i_x] = 0
                    inv_v[i_pb,i_y,i_x] = 0
                    inv_marg_u[i_pb,i_y,i_x] = 0        
                    continue

                # iii. Set d = 0 if the smallest house cannot be afforded
                if x <= par.phi*ph*par.n_min:
                    d[i_pb,i_y,i_x] = 0
                    c[i_pb,i_y,i_x] = linear_interp.interp_1d(grid_m,c_keep[i_pb,i_y,0],x)
                    inv_v[i_pb,i_y,i_x] = linear_interp.interp_1d(grid_m,inv_v_keep[i_pb,i_y,0],x)
                    inv_marg_u[i_pb,i_y,i_x] = 1.0/utility.marg_func_nopar(c[i_pb,i_y,i_x],0,d_ubar,alpha,rho)
                    continue

                # vi. Compute value of each choice of housing
                for i_n in range(par.Nn):
                    d_temp = grid_n[i_n]
                    res = x - par.phi*ph*d_temp # residual cash on hand
                    
                    if res <= 0: # If d_temp cannot be afforded
                        value_of_choice[i_n] = -np.inf
                    else:
                        value_of_choice[i_n] = linear_interp.interp_1d(grid_m,inv_v_keep[i_pb,i_y,i_n],res) # Interpolate over keeper value

                # v. Find optimal choice
                i_n_opt = np.argmax(value_of_choice)
                d[i_pb,i_y,i_x] = grid_n[i_n_opt] # Optimal choice of housing

                # c. Find non-durables and value function
                m = x - par.phi*ph*d[i_pb,i_y,i_x]
                c[i_pb,i_y,i_x] = linear_interp.interp_1d(grid_m,c_keep[i_pb,i_y,i_n_opt],m)
                
                inv_v[i_pb,i_y,i_x] = linear_interp.interp_1d(grid_m,inv_v_keep[i_pb,i_y,i_n_opt],m)    #value_of_choice[i_n_opt]
                inv_marg_u[i_pb,i_y,i_x] = 1.0/utility.marg_func_nopar(c[i_pb,i_y,i_x],d[i_pb,i_y,i_x],d_ubar,alpha,rho)