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
    inv_w = sol.inv_w[t] # sol is an input. This is next periods value func
    q = sol.q[t] # next periods post decision expectation

    # loop over outermost post-decision state
    for i_y in prange(par.Ny):

        # allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        x_plus = np.zeros(par.Na)
        w = np.zeros(par.Na) 
        inv_v_keep_plus = np.zeros(par.Na)
        inv_marg_u_keep_plus = np.zeros(par.Na)
        inv_v_adj_plus = np.zeros(par.Na)
        inv_marg_u_adj_plus = np.zeros(par.Na)
        
        # loop over other outer post-decision states (housing)
        for i_h in range(par.Nh):

            # a. income and durable stock
            y = par.grid_y[i_y]
            h = par.grid_h[i_h]

            # b. initialize at zero
            for i_a in range(par.Na):
                w[i_a] = 0.0
                q[i_y,i_h,i_a] = 0.0

            # c. loop over shocks and then end-of-period assets
            for ishock in range(par.Nshocks): # remove this
                
                # i. shocks
                # psi_plus = par.psi[ishock]
                # psi_plus_w = par.psi_w[ishock]
                # xi_plus = par.xi[ishock]
                # xi_plus_w = par.xi_w[ishock]

                # ii. next-period income and durables
                p_plus = trans.p_plus_func(p,psi_plus,par)
                n_plus = trans.n_plus_func(n,par)

                # iii. prepare interpolators
                prep_keep = linear_interp.interp_3d_prep(par.grid_p,par.grid_n,p_plus,n_plus,par.Na)
                prep_adj = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                # iv. weight
                weight = 1.0 / par.Nshocks # sum to 1

                # v. next-period cash-on-hand and total resources
                for i_a in range(par.Na):
        
                    m_plus[i_a] = trans.m_plus_func(par.grid_a[i_a],p_plus,xi_plus,par)
                    x_plus[i_a] = trans.x_plus_func(m_plus[i_a],n_plus,par)
                
                # vi. interpolate - what is this for
                linear_interp.interp_3d_only_last_vec_mon(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_v_keep[t+1],p_plus,n_plus,m_plus,inv_v_keep_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q:
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_keep,par.grid_p,par.grid_n,par.grid_m,sol.inv_marg_u_keep[t+1],p_plus,n_plus,m_plus,inv_marg_u_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.inv_marg_u_adj[t+1],p_plus,x_plus,inv_marg_u_adj_plus)
                     
                # vii. max and accumulate
                if compute_q:

                    for i_a in range(par.Na):                                

                        keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a] # interpolated values from above. choose optimally
                        if keep:
                            v_plus = -1/inv_v_keep_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_keep_plus[i_a]
                        else:
                            v_plus = -1/inv_v_adj_plus[i_a]
                            marg_u_plus = 1/inv_marg_u_adj_plus[i_a]

                        w[i_a] += weight*par.beta*v_plus
                        q[i_y,i_h,i_a] += weight*par.beta*par.R*marg_u_plus

                else:

                    for i_a in range(par.Na):
                        w[i_a] += weight*par.beta*(-1.0/np.fmax(inv_v_keep_plus[i_a],inv_v_adj_plus[i_a]))
        
            # d. transform post decision value function
            for i_a in range(par.Na):
                inv_w[i_y,i_h,i_a] = -1/w[i_a]