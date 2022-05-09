import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import trans

@njit(parallel=True)
def compute_wq(t,R,sol,par,ph,compute_q=False):
    """ compute the post-decision functions w and q """

    # unpack
    inv_w = sol.inv_w[t]
    q = sol.q[t]

    # a. Loop over outer states
    for i_pb in prange(par.Npb):
        
        # o. Purchase price
        pb = par.grid_pb[i_pb]

        # oo. Income state
        for i_y in prange(par.Ny):

            # allocate temporary containers
            m_plus = np.zeros(par.Na) # container, same lenght as grid_a
            x_plus = np.zeros(par.Na)
            w = np.zeros(par.Na)
            inv_v_keep_plus = np.zeros(par.Na)
            inv_marg_u_keep_plus = np.zeros(par.Na)
            inv_v_adj_plus = np.zeros(par.Na)
            inv_marg_u_adj_plus = np.zeros(par.Na)
            
            # loop over other outer post-decision states
            for i_n in prange(par.Nn): # think of this as the post decision housing

                # a. Income and and housing state
                y = par.grid_y[i_y]
                n = par.grid_n[i_n]

                # b. initialize at zero
                for i_a in range(par.Na):
                    w[i_a] = 0.0
                    q[i_pb,i_y,i_n,i_a] = 0.0

                # c. Loop over income values
                for ishock in range(par.Ny):

                    # ii. Next-period income and durables given ishock
                    y_plus = par.grid_y[ishock]
                    n_plus = trans.n_plus_func(n,par)

                    # iii. Prepare interpolators
                    prep_keep = linear_interp.interp_1d_prep(par.Na) #linear_interp.interp_2d_prep(par.grid_n,n_plus,par.Na)
                    prep_adj = linear_interp.interp_1d_prep(par.Na)

                    # iv. Weight of each income shock from Markov probabilities
                    weight = par.p_mat[i_y,ishock]

                    # v. Next-period cash-on-hand and total resources
                    for i_a in range(par.Na):
                        
                        # o. If keeping next period
                        m_plus[i_a] = trans.m_plus_func(par.grid_a[i_a],y_plus,par,n,R,ph)
                        
                        # oo. If adjusting next period
                        x_plus[i_a] = trans.x_plus_func(m_plus[i_a],n_plus,pb,par,ph) # should this be pb in t+1?
                    
                    # vi. Interpolate
                    linear_interp.interp_1d_vec_mon(prep_keep,par.grid_m,sol.inv_v_keep[t+1,i_pb,ishock,i_n],m_plus,inv_v_keep_plus)
                    linear_interp.interp_1d_vec_mon(prep_adj,par.grid_x,sol.inv_v_adj[t+1,i_pb,ishock],x_plus,inv_v_adj_plus)
                    
                    linear_interp.interp_1d_vec_mon_rep(prep_keep,par.grid_m,sol.inv_marg_u_keep[t+1,i_pb,ishock,i_n],m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_1d_vec_mon_rep(prep_adj,par.grid_x,sol.inv_marg_u_adj[t+1,i_pb,ishock],x_plus,inv_marg_u_adj_plus)
                        
                    # vii. Find max and accumulate.
                    for i_a in range(par.Na):
                        
                        # o. Max over the keeper and adjuster problem
                        keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a]
                        
                        # oo. If keeping is optimal
                        if keep:
                            v_plus = -1/inv_v_keep_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_keep_plus[i_a]
                        
                        # ooo. If adjusting is optimal
                        else:
                            v_plus = -1/inv_v_adj_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_adj_plus[i_a]
                        
                        # oooo. Weight by probabilities
                        w[i_a] += weight*par.beta*v_plus
                        q[i_pb,i_y,i_n,i_a] += weight*par.beta*R*marg_u_plus
            
                # d. transform post decision value function
                for i_a in range(par.Na):
                    inv_w[i_pb,i_y,i_n,i_a] = -1/w[i_a] # Transform to inverse for computations