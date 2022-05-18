from re import S
import numpy as np
from numba import njit, prange
from scipy import interpolate # for intepolation of arrays

# consav
from consav import linear_interp # for linear interpolation
from consav import golden_section_search

# local modules
import utility

#######
# adj #
#######

@njit
def obj_adj(d,x,inv_v_keep,grid_m,ph,par):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x - par.phi*ph*d
    
    # c. value-of-choice
    return linear_interp.interp_1d(grid_m,inv_v_keep,m)  # we are minimizing

# @njit(parallel=True)
def solve_adj(t,sol,par,ph):
    """solve bellman equation for adjusters using nvfi"""

    # unpack output
    inv_v = sol.inv_v_adj[t]
    inv_marg_u = sol.inv_marg_u_adj[t]
    d = sol.d_adj[t]
    c = sol.c_adj[t]

    # unpack input
    inv_v_keep = sol.inv_v_keep[t]
    c_keep = sol.c_keep[t]
    grid_n = par.grid_n
    grid_m = par.grid_m
    d_ubar = par.d_ubar
    alpha = par.alpha
    rho = par.rho

    # Container for value of housing choice
    value_of_choice = np.zeros(shape=par.Nn)

    for i_pb in range(par.Npb):
        for i_y in range(par.Ny):
            
            # loop over x state
            for i_x in range(par.Nx):
                
                # a. Cash-on-hand
                x = par.grid_x[i_x]
                
                if i_x == 0: # Forces c = d = 0
                    d[i_pb,i_y,i_x] = 0
                    c[i_pb,i_y,i_x] = 0
                    inv_v[i_pb,i_y,i_x] = 0
                    inv_marg_u[i_pb,i_y,i_x] = 0        
                    continue

                if x <= par.phi*ph*par.n_min: # If one cannot afford the smallest house
                    d[i_pb,i_y,i_x] = 0
                    c[i_pb,i_y,i_x] = linear_interp.interp_1d(par.grid_m,c_keep[i_pb,i_y,0],x)
                    inv_v[i_pb,i_y,i_x] = linear_interp.interp_1d(par.grid_m,inv_v_keep[i_pb,i_y,0],x)
                    inv_marg_u[i_pb,i_y,i_x] = 1.0/utility.marg_func_nopar(c[i_pb,i_y,i_x],0,d_ubar,alpha,rho)
                    continue

                # b. optimal choice
                for i_n in range(par.Nn):
                    d_temp = par.grid_n[i_n]
                    res = x - par.phi*ph*d_temp # residual cash on hand
                    
                    if res <= 0:
                        value_of_choice[i_n] = -np.inf
                    else:
                        value_of_choice[i_n] = linear_interp.interp_1d(par.grid_m,inv_v_keep[i_pb,i_y,i_n],res)   #obj_adj(d_temp,x,inv_v_keep[i_pb,i_y,i_n,:],grid_m,ph,par)

                i_n_opt = int(np.argmax(value_of_choice))
                d[i_pb,i_y,i_x] = par.grid_n[i_n_opt] # Optimal choice of housing

                # c. optimal value
                m = x - par.phi*ph*d[i_pb,i_y,i_x]
                c[i_pb,i_y,i_x] = linear_interp.interp_1d(par.grid_m,c_keep[i_pb,i_y,i_n_opt],m)
                
                inv_v[i_pb,i_y,i_x] = value_of_choice[i_n_opt] #-obj_adj(d[i_pb,i_y,i_x],x,inv_v_keep[i_pb,i_y,i_n_opt],grid_m,ph,par) # This has to be corrected as well
                inv_marg_u[i_pb,i_y,i_x] = 1.0/utility.marg_func_nopar(c[i_pb,i_y,i_x],d[i_pb,i_y,i_x],d_ubar,alpha,rho)