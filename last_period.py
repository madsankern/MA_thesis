import numpy as np
from numba import njit, prange

# consav
from consav import golden_section_search

# local modules
import utility

# a. objective
@njit
def obj_last_period(d,x,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x - par.ph*d

    return -utility.func(c,d,par)

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
    for i_p in range(par.Np): #prange
        for i_pb in range(par.Npb):
            for i_n in range(par.Nn):
                for i_m in range(par.Nm):
                                
                    # i. states
                    n = par.grid_n[i_n]
                    m = par.grid_m[i_m]

                    if m == 0: # forced c = 0 
                        c_keep[i_pb,i_p,i_n,i_m] = 0
                        inv_v_keep[i_pb,i_p,i_n,i_m] = 0
                        inv_marg_u_keep[i_pb,i_p,i_n,i_m] = 0
                        continue
                    
                    # ii. optimal choice
                    c_keep[i_pb,i_p,i_n,i_m] = m

                    # iii. optimal value
                    v_keep = utility.func(c_keep[i_pb,i_p,i_n,i_m],n,par)
                    inv_v_keep[i_pb,i_p,i_n,i_m] = -1.0/v_keep
                    inv_marg_u_keep[i_pb,i_p,i_n,i_m] = 1.0/utility.marg_func(c_keep[i_pb,i_p,i_n,i_m],n,par)

    # b. adj
    for i_p in range(par.Np): #prange
        for i_pb in range(par.Npb):
            for i_x in range(par.Nx):
                
                # i. states
                x = par.grid_x[i_x]

                if x == 0: # forced c = d = 0
                    d_adj[i_pb,i_p,i_x] = 0
                    c_adj[i_pb,i_p,i_x] = 0
                    inv_v_adj[i_pb,i_p,i_x] = 0
                    inv_marg_u_adj[i_pb,i_p,i_x] = 0
                    continue

                # ii. optimal choices
                d_allow = par.grid_n[par.grid_n <= x/par.ph]
                value_of_choice = np.empty(len(d_allow))
                
                for i_d,d in enumerate(d_allow): # vectorize this loop! , and rename d
                    c = x - par.ph*d
                    value_of_choice[i_d] = utility.func(c,d,par)
                
                d_adj[i_pb,i_p,i_x] = d_allow[np.argmax(value_of_choice)] # written as a max now!
                c_adj[i_pb,i_p,i_x] = x - par.ph*d_adj[i_pb,i_p,i_x]

                # iii. optimal value
                v_adj = -obj_last_period(d_adj[i_pb,i_p,i_x],x,par)
                inv_v_adj[i_pb,i_p,i_x] = -1.0/v_adj
                inv_marg_u_adj[i_pb,i_p,i_x] = 1.0/utility.marg_func(c_adj[i_pb,i_p,i_x],d_adj[i_pb,i_p,i_x],par)