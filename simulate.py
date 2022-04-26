# IMPORTANT: Think about the timing of everything

from joblib import Parallel
import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import markov

# local modules
import trans
import utility

@njit(parallel=True)
def lifecycle(sim,sol,par,path=False):
    """ simulate a panel of households
    
    Args:
        path (bool,optional): simulate along a transition path
     """

    y = sim.y
    n = sim.n
    m = sim.m
    c = sim.c
    d = sim.d
    a = sim.a
    discrete = sim.discrete
    grid_y = par.grid_y
    state = sim.state
    rand = sim.rand
    rand0 = sim.rand0

    # Determine simulation length
    if path:
        horizon = par.path_T
    else:
        horizon = par.sim_T

    # Loop over households and time
    for i in prange(par.simN):
        for t in range(horizon):

            # Determine aggregate states (R and ph)
            if path:
                R = par.path_R[t]
                ph = par.path_ph[t]
            else:
                R = par.R
                ph = par.ph
            
            # # CHANGE THIS BACK AFTER DEBUGGING
            # R = par.path_R[t]
            # ph = par.ph
            
            # a. beginning of period states
            if t == 0:

                # i. Income
                state_lag = markov.choice(rand0[i], par.pi_cum) # Initialize from stationary distribution
                y0 = grid_y[state_lag] # Income in period -1

                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:])
                y[t,i] = grid_y[state[t,i]]
                
                # ii. Housing
                n[t,i] = trans.n_plus_func(sim.d0[i],par)
                
                # iii. Cash on hand
                m[t,i] = trans.m_plus_func(sim.a0[i],y0,par,n[t,i],par.R,par.ph) # Set initial income equal to 'state_lag'

            else:
                # i. Income
                state_lag = state[t-1,i] # last period value
                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:])
                y[t,i] = grid_y[state[t,i]]
                
                # ii. Housing
                n[t,i] = trans.n_plus_func(d[t-1,i],par)
                
                # iii. Cash on hand
                m[t,i] = trans.m_plus_func(a[t-1,i],y[t-1,i],par,n[t,i],par.path_R[t-1],par.path_ph[t-1])
            
            # b. optimal choices and post decision states
            optimal_choice(t,state[t,i],n[t,i],m[t,i],discrete[t,i:],d[t,i:],c[t,i:],a[t,i:],sol,par,ph,path)

@njit
def optimal_choice(t,y,n,m,discrete,d,c,a,sol,par,ph,path): # Calculate the optimal choice

    # No need to iterate over t when simulating in ss
    # if path == False:
    #     t = 0
    t = 0

    # Available cash on hand
    x = trans.x_plus_func(m,n,par.ph,par,ph) # second argument is pb!

    # a. discrete choice
    inv_v_keep = linear_interp.interp_2d(par.grid_n,par.grid_m,sol.inv_v_keep[t,0,y],n,m)
    inv_v_adj = linear_interp.interp_1d(par.grid_x,sol.inv_v_adj[t,0,y],x)
    adjust = inv_v_adj > inv_v_keep
    
    # b. continuous choices
    if adjust:

        discrete[0] = 1 # This is just to compute the share of adjusters
        
        d[0] = linear_interp.interp_1d(
            par.grid_x,sol.d_adj[t,0,y],
            x)

        c[0] = linear_interp.interp_1d(
            par.grid_x,sol.c_adj[t,0,y],
            x)

        tot = ph*d[0]+c[0]
        if tot > x: # Ensure that total consumption only add up to x
            d[0] *= x/tot
            c[0] *= x/tot
            a[0] = 0.0
        else:
            a[0] = x - tot
            
    else:
            
        discrete[0] = 0

        d[0] = n # set housing equal to last period if no adjust

        c[0] = linear_interp.interp_2d(
            par.grid_n,par.grid_m,sol.c_keep[t,0,y],
            n,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]