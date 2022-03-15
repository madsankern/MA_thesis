import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import trans

@njit(parallel=True)
def compute_wq(t,sol,par,compute_q=False):
    """ compute the post-decision functions w and/or q """

    # unpack
    inv_w = sol.inv_w[t]
    q = sol.q[t]

    # loop over outermost post-decision state
    for i_p in prange(par.Np): # Loop over possible values of y here

        # allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        x_plus = np.zeros(par.Na)
        w = np.zeros(par.Na) 
        inv_v_keep_plus = np.zeros(par.Na)
        inv_marg_u_keep_plus = np.zeros(par.Na)
        inv_v_adj_plus = np.zeros(par.Na)
        inv_marg_u_adj_plus = np.zeros(par.Na)
        
        # loop over other outer post-decision states
        for i_n in range(par.Nn): # Loop over state of durables

            # a. permanent income and durable stock
            p = par.grid_p[i_p]
            n = par.grid_n[i_n]

            # b. initialize at zero - why is this necessary?
            for i_a in range(par.Na):
                w[i_a] = 0.0
                q[i_p,i_n,i_a] = 0.0

            # c. loop over shocks and then end-of-period assets
            for ishock in range(len(par.grid_p)): #in range(par.Nshocks): #    # Loop over all shocks to income, weight each shock by probability to compute expectation 
            # This loop might be avoidable

                # i. shocks - can be removed hello
                # psi_plus = par.psi[ishock]
                # psi_plus_w = par.psi_w[ishock]
                # xi_plus = par.xi[ishock]
                # xi_plus_w = par.xi_w[ishock]

                # ii. next-period income and durables
                p_plus = par.grid_p[ishock] #trans.p_plus_func(p,psi_plus,par) # Model for the permament income
                n_plus = trans.n_plus_func(n,par)

                # iii. prepare interpolators - what is this for?
                prep_keep = linear_interp.interp_3d_prep(par.grid_p,par.grid_n,p_plus,n_plus,par.Na)
                prep_adj = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)
                # Note the dimension

                # iv. weight of each income shock
                weight = par.p_mat[i_p,ishock] #1/len(par.grid_p) #par.p_vec([ishock]) #psi_plus_w*xi_plus_w #1.0 / len(par.grid_p) # Add Markov probabilities here.

                # v. next-period cash-on-hand and total resources
                for i_a in range(par.Na):
        
                    m_plus[i_a] = trans.m_plus_func(par.grid_a[i_a],p_plus,par,n)
                    x_plus[i_a] = trans.x_plus_func(m_plus[i_a],n_plus,par) # No need to update this
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_v_keep[t+1],p_plus,n_plus,m_plus,inv_v_keep_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q: # check this
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_marg_u_keep[t+1],p_plus,n_plus,m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.inv_marg_u_adj[t+1],p_plus,x_plus,inv_marg_u_adj_plus)
                     
                # vii. max and accumulate.
                if compute_q:

                    for i_a in range(par.Na):                                
                        # Max over the keeper and adjuster problem
                        keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a]
                        if keep:
                            v_plus = -1/inv_v_keep_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_keep_plus[i_a]
                        else:
                            v_plus = -1/inv_v_adj_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_adj_plus[i_a]
                        # Weight by probabilities
                        w[i_a] += weight*par.beta*v_plus
                        q[i_p,i_n,i_a] += weight*par.beta*par.R*marg_u_plus

                else:
                    # This can be deleted, I think
                    for i_a in range(par.Na):
                        w[i_a] += weight*par.beta*(-1.0/np.fmax(inv_v_keep_plus[i_a],inv_v_adj_plus[i_a]))
        
            # d. transform post decision value function
            for i_a in range(par.Na):
                inv_w[i_p,i_n,i_a] = -1/w[i_a] # Transform to inverse for computations