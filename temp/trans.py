# The law of motion for state variables

import numpy as np
from numba import njit

# Next perod housing
@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = d # no depereciation
    n_plus = np.fmin(n_plus,par.n_max) # upper bound
    return n_plus

# Cash on hand
@njit(fastmath=True)
def m_plus_func(a,p_plus,xi_plus,par):
    y_plus = par.y # just one value right now
    m_plus = par.R*a + y_plus # accrue interests on savings 
    return m_plus

@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par):
    return m_plus + par.p*n_plus # cash on hand + value of sold house