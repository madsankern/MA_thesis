# Simulate a panel along a transition path

from joblib import Parallel
import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import markov

# local modules
import trans
import utility

# Run loop
@njit(parallel=True)
def lifecycle(sim_path,sol_path,par):
    """ Simulate a panel along a transition path """

    # unpack
    p = sim_path.p
    n = sim_path.n
    m = sim_path.m
    c = sim_path.c
    d = sim_path.d
    a = sim_path.a
    discrete = sim_path.discrete
    grid_p = par.grid_p

    state = sim_path.state # Container for income states

    # Draw shocks - move to model.py later
    rand = sim_path.rand

    # Remove markov.choice for prange to work
    for i in prange(par.simN):
        for t in range(par.sim_T + par.path_T):

            # Determine period specific interest rate
            R = par.path_R[t]            

            # a. beginning of period states
            if t == 0:
                # Income
                state_lag = markov.choice(rand[t,i], par.pi_cum) # Initialize from stationary distribution
                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:])
                p[t,i] = grid_p[state[t,i]]
                
                # Housing
                n[t,i] = trans.n_plus_func(sim_path.d0[i],par)
                
                # Cash on hand
                m[t,i] = trans.m_plus_func(sim_path.a0[i],p[t,i],par,n[t,i],R)

            if t > 0:
                # Income
                state_lag = state[t-1,i] # Use last periods value
                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:])
                p[t,i] = grid_p[state[t,i]]
                
                # Housing
                n[t,i] = trans.n_plus_func(d[t-1,i],par)
                
                # Cash on hand
                m[t,i] = trans.m_plus_func(a[t-1,i],p[t,i],par,n[t,i],R)

            # b. optimal choices and post decision states
            optimal_choice(t,state[t,i],n[t,i],m[t,i],discrete[t,i:],d[t,i:],c[t,i:],a[t,i:],sol_path,par)
 
@njit
def optimal_choice(t,p,n,m,discrete,d,c,a,sol_path,par): # Calculate the optimal choice

    # Available cash on hand
    x = trans.x_plus_func(m,n,par.ph,par)

    # a. discrete choice
    inv_v_keep = linear_interp.interp_2d(par.grid_n,par.grid_m,sol_path.inv_v_keep[t,0,p],n,m)
    inv_v_adj = linear_interp.interp_1d(par.grid_x,sol_path.inv_v_adj[t,0,p],x)
    adjust = inv_v_adj > inv_v_keep
    
    # b. continuous choices
    if adjust:

        discrete[0] = 1 # This is just to compute the share of adjusters
        
        d[0] = linear_interp.interp_1d(
            par.grid_x,sol_path.d_adj[t,0,p],
            x)

        c[0] = linear_interp.interp_1d(
            par.grid_x,sol_path.c_adj[t,0,p],
            x)

        tot = par.ph*d[0]+c[0]
        if tot > x: # Ensure that total consumption only add up to x
            d[0] *= x/tot
            c[0] *= x/tot
            a[0] = 0.0
        else:
            a[0] = x - tot
            
    else:
            
        discrete[0] = 0

        d[0] = n

        c[0] = linear_interp.interp_2d(
            par.grid_n,par.grid_m,sol_path.c_keep[t,0,p],
            n,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]