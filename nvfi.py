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

# # Mads' test of generating objective
# @njit
# def obj_adj(x,inv_v_keep,grid_n,grid_m):
#     """ evaluate bellman equation """
#     # Vectorize!

#     # Initialize
#     value_of_choice = np.empty(shape=len(grid_n))

#     # Loop over housing values
#     for i_n,d in enumerate(grid_n):

#         # Cash on hand
#         m = x - d

#         # Durable hosing
#         n = d

#         # Compute value of choice
#         value_of_choice[i_n] = - linear_interp.interp_2d(grid_n,grid_m,inv_v_keep,n,m)

#     return value_of_choice

@njit
def obj_adj(d,x,inv_v_keep,grid_n,grid_m):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x-d

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

    # loop over outer states
    for i_p in range(par.Np): #prange
            
        # loop over x state
        for i_x in range(par.Nx):
            
            # a. cash-on-hand
            x = par.grid_x[i_x]
            # if i_x == 0:
            #     d[i_p,i_x] = 0
            #     c[i_p,i_x] = 0
            #     inv_v[i_p,i_x] = 0
            #     if par.do_marg_u:
            #         inv_marg_u[i_p,i_x] = 0        
            #     continue

            # b. optimal choice

            # d_low = np.fmin(x/2,1e-8)
            # d_high = np.fmin(x,par.n_max)
            # d[i_p,i_x] = golden_section_search.optimizer(obj_adj,d_low,d_high,args=(x,inv_v_keep[i_p],grid_n,grid_m),tol=par.tol)
            
            d_allow = par.grid_n[par.grid_n <= x]
            print(d_allow)
            value_of_choice = np.empty(len(d_allow))

            for i_d,d in enumerate(d_allow): # vectorize this loop!
                
                m = x - d # cash on hand after choosing the durable

                value_of_choice[i_d] = interpolate.interp1d(m,inv_v_keep[i_p,i_d,:]) # Find value of choice by interpolation over inv_v_keep
            
            print(value_of_choice)
            
            d[i_p,i_x] = d_allow[np.argmax(value_of_choice)] #- written as a max now!
            print(d[i_p,i_x])

            
            # c. optimal value
            m = x - d[i_p,i_x]
            c[i_p,i_x] = linear_interp.interp_2d(par.grid_n,par.grid_m,c_keep[i_p],d[i_p,i_x],m)
            inv_v[i_p,i_x] = -obj_adj(d[i_p,i_x],x,inv_v_keep[i_p],grid_n,grid_m) # This has to be corrected as well
            if par.do_marg_u:
                inv_marg_u[i_p,i_x] = 1/utility.marg_func_nopar(c[i_p,i_x],d[i_p,i_x],d_ubar,alpha,rho)