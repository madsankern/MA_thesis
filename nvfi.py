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
def obj_adj(d,x,inv_v_keep,grid_n,grid_m,par):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x - par.ph*d

    # b. durables
    n = d
    
    # c. value-of-choice
    return -linear_interp.interp_2d(grid_n,grid_m,inv_v_keep,n,m)  # we are minimizing


@njit(parallel=True)
def solve_adj(t,sol,par):
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

    # print(inv_v_keep)

    # loop over outer states
    for i_p in prange(par.Np): #prange

        # loop over pb here
        for i_pb in range(par.Npb):
            
            # loop over x state
            for i_x in range(par.Nx):
                
                # a. cash-on-hand - this is fine
                x = par.grid_x[i_x]
                if i_x == 0:
                    d[i_pb,i_p,i_x] = 0
                    c[i_pb,i_p,i_x] = 0
                    inv_v[i_pb,i_p,i_x] = 0
                    if par.do_marg_u: # remove 'if'
                        inv_marg_u[i_pb,i_p,i_x] = 0        
                    continue

                # b. optimal choice
                d_allow = par.grid_n[par.grid_n <= x / par.ph] # House sizes that can be afforded
                value_of_choice = np.empty(len(d_allow)) # Initialize

                for i_d,house in enumerate(d_allow): # vectorize this loop!
                    
                    m_i = x - par.ph*house # cash on hand after choosing the durable
                    value_of_choice[i_d] = linear_interp.interp_1d(par.grid_m,inv_v_keep[i_pb,i_p,i_d,:],m_i) # Find value of choice by interpolation over inv_v_keep
                
                i_opt = np.argmax(value_of_choice) # convert to integer, might not be necessary 
                d_opt = d_allow[i_opt]
                d[i_pb,i_p,i_x] = d_opt # These can be combined

                # c. optimal value
                m = x - par.ph*d[i_pb,i_p,i_x]
                c[i_pb,i_p,i_x] = linear_interp.interp_2d(par.grid_n,par.grid_m,c_keep[i_pb,i_p],d[i_pb,i_p,i_x],m)
                inv_v[i_pb,i_p,i_x] = -obj_adj(d[i_pb,i_p,i_x],x,inv_v_keep[i_pb,i_p],grid_n,grid_m,par) # This has to be corrected as well
                if par.do_marg_u: # This is always the case when using negm
                    inv_marg_u[i_pb,i_p,i_x] = 1/utility.marg_func_nopar(c[i_pb,i_p,i_x],d[i_pb,i_p,i_x],d_ubar,alpha,rho)