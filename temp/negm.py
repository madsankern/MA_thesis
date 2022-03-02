# Solves the keeper problem using the upper envelope extension

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
    c = sol.c_keep[t]
    q_c = sol.q_c[t] # post decision marg. value of cash
    q_m = sol.q_m[t] # check this

    # Loop over income states
    for i_y in prange(par.Ny):
        
        # temporary container
        v_ast_vec = np.zeros(par.Nm)

        # Loop over housing values
        for i_n in range(par.Nn):
            
            # Use Euler equation to find all candidate solutions
            n = par.grid_n[i_n]
            for i_a in range(par.Na):
                q_c[i_y,i_n,i_a] = utility.inv_marg_func(sol.q[t,i_y,i_n,i_a],n,par) # Euler eq. step
                q_m[i_y,i_n,i_a] = par.grid_a[i_a] + q_c[i_y,i_n,i_a] # Endogeneous grid point
        
            # Apply upper envelope step
            negm_upperenvelope(par.grid_a,q_m[i_y,i_n],q_c[i_y,i_n],sol.inv_w[t,i_y,i_n],
               par.grid_m,c[i_y,i_n],v_ast_vec,n,par) # Remove non optimal points on the euler eq.

            # negative inverse - for later calculations (?)
            for i_m in range(par.Nm):
                inv_v[i_y,i_n,i_m] = -1/v_ast_vec[i_m]
                if par.do_marg_u:
                    inv_marg_u[i_y,i_n,i_m] = 1/utility.marg_func(c[i_y,i_n,i_m],n,par)