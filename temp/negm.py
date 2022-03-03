# Solves the keeper problem using the upper envelope extension
# Rename to 'consumption problem' or combine the negm and vfi step into one file.

import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope # Upper envelope

# local modules
import utility

# Generate upper envelope given the household utility function
negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)

# Solver function
@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack
    inv_v = sol.inv_v_keep[t]
    inv_marg_u = sol.inv_marg_u_keep[t]
    c = sol.c_keep[t] # Next periods optimal choice for keeper
    q_c = sol.q_c[t] # post decision marg. value of cash
    q_m = sol.q_m[t] # check this

    # Loop over income states
    for i_y in prange(par.Ny):
        
        # temporary container
        v_ast_vec = np.zeros(par.Nm)

        # Loop over housing values
        for i_h in range(par.Nh):
        # additionally, loop over house prices
            
            # Set housing as value in loop
            h = par.grid_h[i_h]

            # Grid over post decision cash
            for i_a in range(par.Na):

                # q is what you think it is (from the paper)
                q_c[i_y,i_h,i_a] = utility.inv_marg_func(sol.q[t,i_y,i_h,i_a],h,par) # Euler eq. step         
                q_m[i_y,i_h,i_a] = par.grid_a[i_a] + q_c[i_y,i_h,i_a] # Endogeneous grid point
                # Think about what q_c and q_m mean

            # Apply upper envelope step
            negm_upperenvelope(par.grid_a,q_m[i_y,i_h],q_c[i_y,i_h],sol.inv_w[t,i_y,i_h],
               par.grid_m,c[i_y,i_h],v_ast_vec,h,par) # Remove non optimal points on the euler eq.

            # negative inverse - for later calculations
            for i_m in range(par.Nm):
                inv_v[i_y,i_h,i_m] = -1/v_ast_vec[i_m] # inverse value at the endogenous state
                if par.do_marg_u:
                    inv_marg_u[i_y,i_h,i_m] = 1/utility.marg_func(c[i_y,i_h,i_m],h,par) # inverse marginal utility