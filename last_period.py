import numpy as np
from numba import njit, prange

# consav
from consav import golden_section_search

# local modules
import utility

# a. objective
@njit
def obj_last_period(d,x,par,ph):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x - par.phi*ph*d

    return utility.func(c,d,par)

#@njit #(parallel=True)
def solve(t,sol,par,ph):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep[t]
    inv_marg_u_keep = sol.inv_marg_u_keep[t]
    c_keep = sol.c_keep[t]
    inv_v_adj = sol.inv_v_adj[t]
    inv_marg_u_adj = sol.inv_marg_u_adj[t]
    d_adj = sol.d_adj[t]
    c_adj = sol.c_adj[t]

    value_of_choice = np.zeros(shape=par.Nn) # for choosing optimal housing

    # a. keep
    for i_y in range(par.Ny):
        for i_pb in range(par.Npb): # no need to loop over this
            for i_n in range(par.Nn):
                for i_m in range(par.Nm):
                                
                    # i. states
                    n = par.grid_n[i_n]
                    m = par.grid_m[i_m]

                    if m == 0: # forced c = 0 
                        c_keep[i_pb,i_y,i_n,i_m] = 0
                        inv_v_keep[i_pb,i_y,i_n,i_m] = 0
                        inv_marg_u_keep[i_pb,i_y,i_n,i_m] = 0
                        continue
                    
                    # ii. optimal choice
                    c_keep[i_pb,i_y,i_n,i_m] = m

                    # iii. optimal value
                    v_keep = utility.func(c_keep[i_pb,i_y,i_n,i_m],n,par)
                    inv_v_keep[i_pb,i_y,i_n,i_m] = -1.0/v_keep
                    inv_marg_u_keep[i_pb,i_y,i_n,i_m] = 1.0/utility.marg_func(c_keep[i_pb,i_y,i_n,i_m],n,par)

    # b. adj
    for i_y in range(par.Ny): #prange
        for i_pb in range(par.Npb):
            for i_x in range(par.Nx):
                
                # i. States
                x = par.grid_x[i_x]

                if x == 0: # forced c = d = 0
                    d_adj[i_pb,i_y,i_x] = 0
                    c_adj[i_pb,i_y,i_x] = 0
                    inv_v_adj[i_pb,i_y,i_x] = 0
                    inv_marg_u_adj[i_pb,i_y,i_x] = 0
                    continue
                
                if x <= par.phi*ph*par.n_min: # If one cannot afford the smallest house
                    d_adj[i_pb,i_y,i_x] = 0
                    c_adj[i_pb,i_y,i_x] = x
                    inv_v_adj[i_pb,i_y,i_x] = -1.0/utility.func(c_adj[i_pb,i_y,i_x],0,par)
                    inv_marg_u_adj[i_pb,i_y,i_x] = 1.0/utility.marg_func(c_adj[i_pb,i_y,i_x],0,par)
                    continue

                # ii. Optimal choices
                for i_n in range(par.Nn):
                    d = par.grid_n[i_n]
                    c = x - par.phi*ph*d
                    
                    if c <= 0:
                        value_of_choice[i_n] = -np.inf
                    else:
                        value_of_choice[i_n] = -1.0/utility.func(c,d,par)

                i_opt = np.argmax(value_of_choice)
                # print(value_of_choice)
                # print(i_opt)
                d_adj[i_pb,i_y,i_x] = par.grid_n[i_opt] # Optimal choice of housing
                c_adj[i_pb,i_y,i_x] = x - par.phi*ph*d_adj[i_pb,i_y,i_x] # use residual income on the non-durable

                # iii. optimal value
                v_adj = utility.func(c_adj[i_pb,i_y,i_x],d_adj[i_pb,i_y,i_x],par) #obj_last_period(d_adj[i_pb,i_y,i_x],x,par,ph)
                inv_v_adj[i_pb,i_y,i_x] = -1.0/v_adj
                inv_marg_u_adj[i_pb,i_y,i_x] = 1.0/utility.marg_func(c_adj[i_pb,i_y,i_x],d_adj[i_pb,i_y,i_x],par)