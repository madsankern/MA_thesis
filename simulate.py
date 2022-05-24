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
def monte_carlo(sim,sol,par,path=False):
    """ Simulate a panel of households using Monte Carlo
    
    Args:
        path (bool,optional): Simulate along a transition path
     """

    y = sim.y
    n = sim.n
    m = sim.m
    c = sim.c
    d = sim.d
    a = sim.a
    pb = sim.pb
    discrete = sim.discrete
    grid_y = par.grid_y
    state = sim.state
    rand = sim.rand
    rand0 = sim.rand0

    # a. Determine simulation length
    if path:
        horizon = par.path_T
    else:
        horizon = par.sim_T

    # b. Loop over households and time
    for i in prange(par.simN):
        for t in range(horizon):

            # i. Determine aggregate states (R and ph)
            if path:
                R = par.path_R[t]
                ph = par.path_ph[t]
            else:
                R = par.R
                ph = par.ph
            
            # ii. Determine beginning of period states
            if t == 0:

                # o. Purchase price
                pb_lag = par.ph

                # oo. Income
                if path:
                    state_lag = int(sim.state_lag[i]) #markov.choice(rand0[i], par.pi_cum) #0.0 #state_lagg[i] #sim.state_lag
                else:
                    state_lag = markov.choice(rand0[i], par.pi_cum) # Initialize from stationary distribution

                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:]) # Finds the INDEX of current y
                y[t,i] = grid_y[state[t,i]] # Income in period t
                
                # ooo. Housing
                n[t,i] = trans.n_plus_func(sim.d0[i],par)
                
                # oooo. Cash on hand
                m[t,i] = trans.m_plus_func(sim.a0[i],par.R,y[t,i],n[t,i],par.ph,par) # Uses ss prices

            else:

                # oooo. Lagged purchase price
                pb_lag = pb[t-1,i]

                # o. Income
                state_lag = state[t-1,i] # last period value

                state[t,i] = markov.choice(rand[t,i], par.p_mat_cum[state_lag,:])
                y[t,i] = grid_y[state[t,i]]
                
                # oo. Housing
                n[t,i] = trans.n_plus_func(d[t-1,i],par)
                
                # ooo. Cash on hand
                m[t,i] = trans.m_plus_func(a[t-1,i],par.path_R[t-1],y[t,i],n[t,i], par.path_ph[t-1],par)

            
            # b. Find optimal choices and post decision states
            optimal_choice(t,state[t,i],n[t,i],m[t,i],pb_lag,ph,discrete[t,i:],d[t,i:],c[t,i:],a[t,i:],pb[t,i:],sol,par,path)

@njit
def optimal_choice(t,state,n,m,pb_lag,ph,discrete,d,c,a,pb,sol,par,path=False):
# order of input: beggining of period states, end period states,containers,option

    # a. Ensure t is constant if simulating in steady state
    if path == False:
        t = 0

    # b. Available cash-on-hand if adjusting
    x = trans.x_plus_func(m,n,ph,pb_lag,par)
    
    # c. House purchase price if either adjusting or keeping
    pb_adj = ph
    pb_keep = pb_lag

    # d. Find max of keeper and adjuster (discrete choice)
    inv_v_keep = linear_interp.interp_3d(par.grid_pb,par.grid_n,par.grid_m,sol.inv_v_keep[t,:,state],pb_keep,n,m)
    inv_v_adj = linear_interp.interp_2d(par.grid_pb,par.grid_x,sol.inv_v_adj[t,:,state],pb_adj,x)
    adjust = inv_v_adj > inv_v_keep
    
    # b. Find implied durable and non-durable consumption from discrete choice
    if adjust:

        discrete[0] = 1 
        pb[0] = pb_adj # Update purchase price
        
        d[0] = linear_interp.interp_2d(
            par.grid_pb,par.grid_x,sol.d_adj[t,:,state],
            pb_adj,x)

        c[0] = linear_interp.interp_2d(
            par.grid_pb,par.grid_x,sol.c_adj[t,:,state],
            pb_adj,x)

        tot = par.phi*ph*d[0]+c[0]
        if tot > x: # Ensure that total consumption only add up to x
            # d[0] *= x/tot
            # c[0] *= x/tot
            c[0] -= tot-x
            a[0] = 0.0
            # discrete[0] = 2
        else:
            a[0] = x - tot
                            
    else:
            
        discrete[0] = 0
        pb[0] = pb_keep

        d[0] = n # set housing equal to last period if no adjust

        c[0] = linear_interp.interp_3d(
            par.grid_pb,par.grid_n,par.grid_m,sol.c_keep[t,:,state],
            pb_keep,n,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]